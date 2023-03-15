from typing import List
from typing import Union
import numpy as np
from skimage import transform
from skimage import registration
from skimage import feature
from scipy.ndimage.interpolation import map_coordinates

import cv2
import os
import tifffile
from scipy import ndimage
from merlin.core import analysistask
from merlin.util import aberration
from merlin.util import imagewriter

class EstimateTissueThickness(analysistask.ParallelAnalysisTask):

    """
    An abstract class for estimating the tissue thickness based
    on the beads imaging.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if "highpass_sigma" not in self.parameters:
            self.parameters['highpass_sigma'] = -1

        if "median_filter_size" not in self.parameters:
            self.parameters['median_filter_size'] = 2

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        return self._high_pass_filter(self._median_filter(inputImage))

    def _filter_set(self, inputImages: np.ndarray) -> np.ndarray:
        return np.array([ self._high_pass_filter(self._median_filter(x)) \
            for x in inputImages ])
    
    def _median_filter(self, inputImage: np.ndarray) -> np.ndarray:
        median_filter_size = self.parameters['median_filter_size']
        return ndimage.median_filter(inputImage, 
            size=median_filter_size, mode="mirror")
    
    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters['highpass_sigma']
        if highPassSigma == -1:
            return inputImage
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
        return inputImage.astype(float) - cv2.GaussianBlur(
                inputImage, (highPassFilterSize, highPassFilterSize),
                highPassSigma, borderType=cv2.BORDER_REPLICATE) 
        
    def _save_thickness(self, transformationList, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='thickness')
    
    def get_feature_image_set(self, dataChannel, fov: int):
        return np.array([ self.dataSet.get_feature_image(dataChannel, fov, zpos) \
            for zpos in self.dataSet.get_data_organization().get_feature_z_positions() ])
    
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []


    def _run_analysis(self, fragmentIndex: int):
        intensity_z_list = []
        for dataChannel in self.dataSet.get_data_organization().get_data_channels():
            images = self.get_feature_image_set(
                    dataChannel, fragmentIndex)
            imagesFiltered = self._filter_set(images)
            intensity_z_list.append(
                imagesFiltered.sum(axis=1).sum(axis=1).argmax())

        self._save_thickness(intensity_z_list, fragmentIndex)
    
class Interpolate3D(analysistask.ParallelAnalysisTask):

    """
    An abstract class for interpolating 3D image stack
    between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if "fixed_channel" not in self.parameters:
            self.parameters['fixed_channel'] = 0
        
        if "channel_names" not in self.parameters:
            self.parameters['channel_names'] = [ \
                sel.get_data_channel_name(x) \
                    for x in self.dataSet.get_data_channels() ]

        if "iterative_registration" not in self.parameters:
            self.parameters['iterative_registration'] = False
        
        if "write_aligned_images" not in self.parameters:
            self.parameters['write_aligned_images'] = False

        if "write_aligned_features" not in self.parameters:
            self.parameters['write_aligned_features'] = False
            
        if "image_z_pixel_size_micron" not in self.parameters:
            self.parameters['image_z_pixel_size_micron'] = 0.5

        if "highpass_sigma" not in self.parameters:
            self.parameters['highpass_sigma'] = -1

        if "median_filter_size" not in self.parameters:
            self.parameters['median_filter_size'] = 2

        if "file_type" not in self.parameters:
            self.parameters['file_type'] = "dax"

        if "interpolate_thickness_micron" not in self.parameters:
            self.parameters['interpolate_thickness_micron'] = 3

        if "max_depth_index" not in self.parameters:
            self.parameters['max_depth_index'] = 100
                    
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def interpolate_single_image(self, movie, zIndex, shifts):
        # get the 3D coordinates
        single_im_size = np.array([movie.shape[1], movie.shape[2]])
        _coords = np.meshgrid( 
            np.arange(single_im_size[0]), 
            np.arange(single_im_size[1]),)
    
        _coords = np.stack(_coords).transpose((0, 2, 1)) # transpose is necessary 
        _coords = _coords.reshape(_coords.shape[0], -1)
        _coords_3D = np.zeros([3, _coords.shape[1]])
        _coords_3D[0,:] = zIndex - shifts[0]
        _coords_3D[1,:] = _coords[0,:] - shifts[1]
        _coords_3D[2,:] = _coords[1,:] - shifts[2]
    
        _corr_im = map_coordinates(movie, _coords_3D, order=1)
        return _corr_im.reshape(single_im_size).astype(movie.dtype)

    def get_shift(self, fov: int, dataChannel: int=None):
        shifts = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='shifts')

        if dataChannel is not None:
            return shifts[dataChannel]
        else:
            return shifts

    def get_interpolated_feature_set(
            self, fov: int, 
            dataChannel: int, 
            ) -> np.ndarray:

        """Get the interpolated feature images

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zPosList: a list of z positions in micron to be interpolated
        Returns:
            a 3-dimensional numpy array containing the interpolated image set
        """
    
        zPixelSizeMicron = self.parameters['feature_z_pixel_size_micron']
        zmax = max(self.dataSet.get_feature_z_positions()) + \
            self.parameters['interpolate_thickness_micron']
        
        movie = np.zeros([
            int(zmax / zPixelSizeMicron) + 1, 
            self.parameters['feature_dimentions'][0],
            self.parameters['feature_dimentions'][1]])

        for z in self.dataSet.get_feature_z_positions():
            movie[int(z / zPixelSizeMicron)] = \
                self.dataSet.get_feature_image(dataChannel, fov, z)

        offset = self.get_transform(fov, dataChannel)
        transformations_xy = transform.SimilarityTransform(
            translation=[-offset[2], -offset[1]]) 
        
        movie = np.array([ 
            transform.warp(x, transformations_xy, preserve_range=True).astype(movie.dtype) 
            for x in movie ])
        
        images_res = []
        for zPos in np.array(self.parameters['feature_z_coordinates_micron']):
            imgZmax = np.array([ self.interpolate_single_image(
                movie = movie, 
                zIndex = z / zPixelSizeMicron, 
                shifts = self.get_shift(fov, dataChannel) * z) \
                    for z in np.arange(
                        zPos, 
                        zPos + self.parameters['interpolate_thickness_micron'], 
                        zPixelSizeMicron) ]
                ).max(axis=0)
            images_res.append(imgZmax)
        
        return images_res
        
    def get_interpolated_image_set(
            self, fov: int, 
            dataChannel: int, 
            ) -> np.ndarray:

        """Get the interpolated image set

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
        Returns:
            a 3-dimensional numpy array containing the interpolated image set
        """
    
        zPixelSizeMicron = self.parameters['image_z_pixel_size_micron'];
        zmax = max(self.dataSet.get_z_positions());
        
        movie = np.zeros([
            int(zmax / zPixelSizeMicron) + 1, 
            self.parameters['image_dimentions'][0],
            self.parameters['image_dimentions'][1]]);

        for z in self.dataSet.get_z_positions():
            movie[int(z / zPixelSizeMicron)] = \
                self.dataSet.get_raw_image(dataChannel, fov, z);
        
        offset = self.get_transform(fov, dataChannel)
        transformations_xy = transform.SimilarityTransform(
            translation=[-offset[2], -offset[1]]) 
        
        movie = np.array([ 
            transform.warp(x, transformations_xy, preserve_range=True).astype(movie.dtype) 
            for x in movie ])

        images_res = []
        for zPos in np.array(self.parameters['image_z_coordinates_micron']):
            imgZmax = np.array([ self.interpolate_single_image(
                movie = movie, 
                zIndex = z / zPixelSizeMicron, 
                shifts = self.get_shift(fov, dataChannel) * z) \
                    for z in np.arange(
                        zPos, 
                        zPos + self.parameters['interpolate_thickness_micron'], 
                        zPixelSizeMicron) ]
                ).max(axis=0)
            images_res.append(imgZmax)
        
        return images_res
        
    def get_transformation(self, fov: int, dataChannel: int=None
                            ) ->np.ndarray:
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        if dataChannel is not None:
            return transformationMatrices[dataChannel]
        else:
            return transformationMatrices

    def get_shift_pixel(self, fov: int, dataChannel: int=None):
        shifts = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='shifts')
        if dataChannel is not None:
            return shifts[dataChannel]
        else:
            return shifts

    def get_transform(self, fov: int, dataChannel: int=None):
        shifts = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        if dataChannel is not None:
            return shifts[dataChannel]
        else:
            return shifts

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        return self._high_pass_filter(self._median_filter(inputImage))

    def _filter_set(self, inputImages: np.ndarray) -> np.ndarray:
        return np.array([ self._high_pass_filter(self._median_filter(x)) \
            for x in inputImages ])

    def _median_filter(self, inputImage: np.ndarray) -> np.ndarray:
        median_filter_size = self.parameters['median_filter_size']
        return ndimage.median_filter(inputImage, 
            size=median_filter_size, mode="mirror")
    
    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters['highpass_sigma']
        if highPassSigma == -1:
            return inputImage
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
        return inputImage.astype(float) - cv2.GaussianBlur(
                inputImage, (highPassFilterSize, highPassFilterSize),
                highPassSigma, borderType=cv2.BORDER_REPLICATE) 

    def get_feature_image_set(self, dataChannel, fov: int):
        return np.array([ self.dataSet.get_feature_image(dataChannel, fov, zpos) \
            for zpos in self.dataSet.get_data_organization().get_feature_z_positions() ])

    def _save_transformations(self, transformationList, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')

    def _save_shifts(self, shiftList, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(shiftList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='shifts')
    
    def _analysis_image_name(self,
                             subdirectory: str,
                             imageBaseName: str, 
                             imageIndex: int,
                             dataChannel: int,
                             fileType = "tif"
                             ) -> str:
        destPath = self.dataSet.get_analysis_subdirectory(
                self, subdirectory=subdirectory)
        return os.sep.join([destPath, 
            imageBaseName+"_"+str(imageIndex)+"_"+str(dataChannel)+'.'+fileType])

    def writer_for_analysis_data(self,
                                 data: np.ndarray,
                                 subdirectory: str,
                                 imageBaseName: str,
                                 imageIndex: int,
                                 dataChannel: int,
                                 fileType = 'dax'
                                 ) -> None:

        fname = self._analysis_image_name(
                subdirectory, imageBaseName,
                imageIndex, dataChannel, 
                fileType)
        
        if fileType == "dax":
            f = imagewriter.DaxWriter(fname)
            for x in data.astype(np.uint16):
                f.addFrame(x)
            f.close()
        elif fileType == "tif":
            tifffile.imwrite(
                data=data.astype(np.uint16),
                file=fname)
        elif fileType == "npy":
            np.save(fname, data)

    def _run_analysis(self, fragmentIndex: int):

        try:
            shifts3D = self.get_shift_pixel(fragmentIndex);
            shifts2D = self.get_transformation(fragmentIndex);
        except (FileNotFoundError, OSError, ValueError):
            
            dataChannels = self.dataSet.\
                get_data_organization().get_data_channels()

            fixImage = np.array([self._filter(
                self.dataSet.get_feature_fiducial_image(
                    self.parameters['fixed_channel'], 
                    fragmentIndex)) ]);

            shifts2D = np.array([registration.phase_cross_correlation(
                reference_image = fixImage,
                moving_image = np.array([self._filter(
                    self.dataSet.get_feature_fiducial_image(
                        x, fragmentIndex)) ]),
                upsample_factor = 100)[0] for x in \
                    dataChannels ])
            
            self._save_transformations(
                shifts2D, fragmentIndex)

            # get fixed image stack
            fixImageStack = self.get_feature_image_set(
                    self.parameters['fixed_channel'], 
                    fragmentIndex)

            # get max intensity frame before filtering
            maxIndex = self.parameters['max_depth_index']
            frameZposList = self.dataSet.dataOrganization.get_feature_z_positions()
            maxDepth = frameZposList[maxIndex]
            
            # filter the images
            fixImageStack = self._filter_set(fixImageStack)
            
            # block the beads info outside the surface
            fixImageStack[:(maxIndex-10),:,:] = 0
            fixImageStack[(maxIndex+10):,:,:] = 0
            
            shifts3D = np.array([
                registration.phase_cross_correlation(
                    reference_image = fixImageStack,
                    moving_image = self._filter_set(
                        self.get_feature_image_set(x, fragmentIndex)),
                    upsample_factor = 100)[0] \
                    for x in dataChannels ])

            if self.parameters['iterative_registration']:
                # generate max index frame
                maxIndexs = (np.array([maxIndex] * shifts3D.shape[0]) + \
                    shifts3D[:,0]).astype(np.uint16)

                # extract frames only has the beads
                fixImageStack[:(maxIndexs[self.parameters['fixed_channel']]-10),:,:] = 0
                fixImageStack[(maxIndexs[self.parameters['fixed_channel']]+10):,:,:] = 0
                
                # fine estimation of the shift
                shifts3D = [];
                for ri in self.dataSet.get_data_organization().get_data_channels():
                    movImageStack = self._filter_set(
                        self.get_feature_image_set(ri, fragmentIndex))
                
                    movImageStack[:(maxIndexs[ri]-10),:,:] = 0
                    movImageStack[(maxIndexs[ri]+10):,:,:] = 0
                    
                    r = registration.phase_cross_correlation(
                        fixImageStack, movImageStack, 
                        upsample_factor=100)

                    shifts3D.append(r[0])
                shifts3D = np.array(shifts3D)

            self._save_shifts(
                (shifts3D - shifts2D) / maxDepth, 
                fragmentIndex);

        if self.parameters['write_aligned_features']:
            dataChannels = [ self.dataSet.get_data_organization().\
                    get_data_channel_index(x) \
                for x in self.parameters['channel_names'] ]

            for dataChannel in dataChannels:
                imageSet = self.get_interpolated_feature_set(
                        fragmentIndex, dataChannel) 
                
                fimage = self.dataSet.get_feature_fiducial_image(
                    dataChannel, fragmentIndex)
                
                offset = self.get_transform(fragmentIndex, dataChannel)
                transformations_xy = transform.SimilarityTransform(
                    translation=[-offset[2], -offset[1]]) 
        
                fimage = transform.warp(fimage, 
                                        transformations_xy, 
                                        preserve_range=True
                                        ).astype(fimage.dtype) 

                self.writer_for_analysis_data(
                    np.array([fimage] + imageSet), 
                    subdirectory = "interpolatedFeatures",
                    imageBaseName = "images", 
                    imageIndex = fragmentIndex,
                    dataChannel = dataChannel,
                    fileType = self.parameters['file_type'])

        # write down interpolated images
        if self.parameters['write_aligned_images']:
            dataChannels = [ self.dataSet.get_data_organization().\
                    get_data_channel_index(x) \
                for x in self.parameters['channel_names'] ]

            for dataChannel in dataChannels:
                imageSet = self.get_interpolated_image_set(
                        fragmentIndex, dataChannel) 

                fimage = self.dataSet.get_fiducial_image(
                    dataChannel, fragmentIndex)
                                
                offset = self.get_transform(fragmentIndex, dataChannel)
                transformations_xy = transform.SimilarityTransform(
                    translation=[-offset[2], -offset[1]]) 
        
                fimage = transform.warp(fimage, 
                                        transformations_xy, 
                                        preserve_range=True
                                        ).astype(fimage.dtype) 
                
                self.writer_for_analysis_data(
                    np.array([fimage] + imageSet), 
                    subdirectory = "interpolatedImages",
                    imageBaseName = "images", 
                    imageIndex = fragmentIndex,
                    dataChannel = dataChannel,
                    fileType = self.parameters['file_type'])
        

