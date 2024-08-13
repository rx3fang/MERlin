import os
import cv2
import numpy as np
from typing import Dict
from typing import List
from skimage import transform

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.util import imagewriter
from merlin.data import codebook


class Preprocess(analysistask.ParallelAnalysisTask):

    """
    An abstract class for preparing data for barcode calling.
    """

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])

    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result(
                'pixel_histogram', self.analysisName, fov, 'histograms')

        pixelHistogram = np.zeros(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(
            histogram, 'pixel_histogram', self.analysisName, fov, 'histograms')

class DeconvolutionPreprocess(Preprocess):
    
    """
    Perform deconvolution to MERFISH images.
    
        Parameters:
        -----------
        highpass_sigma : float
            High pass filter sigma to remove cellular background
        
        decon_sigma : int
            Deconvolution sigma.
        
        decon_filter_size : float
            Deconvolution filter size.

        decon_iterations : float
            Deconvolution iteration.

        codebook_index : int
            The codebook to be used. Sometime, we decode two codebooks at the same time,
            this parameter determines which codebook to be used.
        
        save_pixel_histogram: bool
            whether to save the histogram of the pixel intenisty. 
    
    Rongxin Fang
    7/31/2024
    """
    
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        # disable deconvolution
        if 'decon_sigma' not in self.parameters:
            self.parameters['decon_sigma'] = -1
        if 'decon_filter_size' not in self.parameters:
            self.parameters['decon_filter_size'] = \
                int(2 * np.ceil(2 * self.parameters['decon_sigma']) + 1)
        if 'decon_iterations' not in self.parameters:
            self.parameters['decon_iterations'] = 20
        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0
        if 'save_pixel_histogram' not in self.parameters:
            self.parameters['save_pixel_histogram'] = False
        
        self._highPassSigma = self.parameters['highpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])
    
    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])
    
    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                     chromaticCorrector)
        
        imageColor = self.dataSet.get_data_organization()\
                        .get_data_channel_color(dataChannel)
        
        return self._preprocess_image(inputImage, imageColor)
    
    def _preprocess_image(
            self, inputImage: np.ndarray, imageColor: str
    ) -> np.ndarray:
        # adjust the image illumination using dark stage
        # and flat field image
        
        imageDark = self.dataSet.illuminationCorrections[imageColor]["dark"]
        imageFlat = self.dataSet.illuminationCorrections[imageColor]["flat"]
        correctedImage = (inputImage - imageDark) / imageFlat

        # make sure values of the corrected images are not negative
        correctedImage = np.clip(correctedImage, 
            a_min=0, a_max=correctedImage.max())

        # high pass filter to remove background
        if self._highPassSigma != -1:
            filteredImage = self._high_pass_filter(
                correctedImage)
        else:
            filteredImage = correctedImage
            
        # deconvolution (disabled when _deconSigma is -1)
        deconFilterSize = self.parameters['decon_filter_size']
        if self._deconSigma == -1:
            deconvolvedImage = filteredImage.astype(np.uint16)
        else:
            deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
                filteredImage, deconFilterSize, self._deconSigma,
                self._deconIterations).astype(np.uint16)
        return deconvolvedImage

    def _high_pass_filter(self, inputImage: np.ndarray, 
            _highPassSigma: int = 2) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * _highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                _highPassSigma)
        return hpImage.astype(np.float)
    
    def _run_analysis(self, fragmentIndex):
            pass

class ImageEnhanceProcess(Preprocess):
    
    """
    Image enhancement by CSB deep learning.
        
    Rongxin Fang
    7/30/2024
    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 2
        # disable deconvolution
        if 'decon_sigma' not in self.parameters:
            self.parameters['decon_sigma'] = -1
        if 'decon_filter_size' not in self.parameters:
            self.parameters['decon_filter_size'] = \
                int(2 * np.ceil(2 * self.parameters['decon_sigma']) + 1)
        if 'decon_iterations' not in self.parameters:
            self.parameters['decon_iterations'] = 20
        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0
        if 'save_pixel_histogram' not in self.parameters:
            self.parameters['save_pixel_histogram'] = False
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1.0

        self._highPassSigma = self.parameters['highpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
    
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])
    
    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])
    
    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                     chromaticCorrector)
        
        imageColor = self.dataSet.get_data_organization()\
                        .get_data_channel_color(dataChannel)
        
        return self._preprocess_image(inputImage, imageColor)
    
    def _preprocess_image(
            self, inputImage: np.ndarray, imageColor: str
    ) -> np.ndarray:
        # adjust the image illumination using dark stage
        # and flat field image
        
        imageDark = self.dataSet.illuminationCorrections[imageColor]["dark"]
        imageFlat = self.dataSet.illuminationCorrections[imageColor]["flat"]
        correctedImage = (inputImage - imageDark) / imageFlat

        # make sure values of the corrected images are not negative
        correctedImage = np.clip(correctedImage, 
            a_min=0, a_max=correctedImage.max())

        # enhance image quality using pre-trained deep learning model
        predictedImage = self._predict(correctedImage, 
            imageColor = imageColor)

        # high pass filter to remove cellular background
        if self._highPassSigma != -1:
            filteredImage = self._high_pass_filter(predictedImage, 
                _highPassSigma = self._highPassSigma)
        else:
            filteredImage = predictedImage

        # deconvolution (disabled when _deconSigma is -1)
        deconFilterSize = self.parameters['decon_filter_size']
        if self._deconSigma == -1:
            deconvolvedImage = filteredImage.astype(np.uint16)
        else:
            deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
                filteredImage, deconFilterSize, self._deconSigma,
                self._deconIterations).astype(np.uint16)
        return deconvolvedImage

    def _high_pass_filter(self, inputImage: np.ndarray, 
            _highPassSigma: int = 2) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * _highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                _highPassSigma)
        return hpImage.astype(np.float)

    def _predict(self, inputImage: np.ndarray, 
                 imageColor: str) -> np.ndarray:
        
        from csbdeep.models import Config, CARE
        self.dataSet._load_deepmerfish_model()

        imageSize = inputImage.shape
        predictedImage = self.dataSet.deepmerfishModel[imageColor].keras_model.predict(
            inputImage.reshape(1, imageSize[0], imageSize[1], 1))
        predictedImage = predictedImage.reshape(imageSize[0], imageSize[1])
        return np.where(predictedImage < 0, 0, predictedImage)
        
    def _run_analysis(self, fragmentIndex):
        pass        
    
class WriteDownProcessedImages(analysistask.ParallelAnalysisTask):

    """
    Writes down the preprocessed images prior to barcode calling.
    
    Parameters:
    -----------
    lowpass_sigma : float
        Low pass filter to smooth the images after deconvolution.
        
    warp_task : str
        The warp task that aligns images from different rounds.
        
    preprocess_task : str
        The preprocessing task.
        
    Optional Parameters:
    ---------------------
    feature_channels : list of str, optional
        The channel names for immunostaining or other cellular features such as DAPI, polyT.
        This can be left empty.
        
    optimize_task : str, optional
        The optimization task. If provided, the images will also be corrected for chromatic aberration, normalized for intensity differences, etc.
    
    Rongxin Fang
    7/31/2024
    """
    
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1.0
        if 'feature_channels' not in self.parameters:
            self.parameters['feature_channels'] = []
        if 'file_type' not in self.parameters:
            self.parameters['file_type'] = 'tif'
        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        self.preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5
        
    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['preprocess_task']]
    
    def get_codebook(self) -> codebook.Codebook:
        return self.preprocessTask.get_codebook()
    
    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})
    
    def _get_initial_chromatic_corrector(self):
        usedColors = self._get_used_colors()
        return {u: {v: transform.SimilarityTransform()
            for v in usedColors if v >= u} for u in usedColors}
    
    def _get_reference_color(self):
        return min(self._get_used_colors())
    
    def get_feature_image_set(
            self, fov, zIndex, featureChannels,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if len(featureChannels) == 0:
            return None
        return np.array([self.warpTask.get_aligned_image(
            fov, self.dataSet.get_data_organization()
                .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for b in featureChannels ])
    
    def _run_analysis(self, fragmentIndex):

        # if optimization task is given, load scalefactor, chromatic abberation
        if 'optimize_task' in self.parameters:
            optimizeTask = self.dataSet.load_analysis_task(
            		self.parameters['optimize_task'])
            scaleFactors = optimizeTask.get_scale_factors() 
            backgrounds = optimizeTask.get_backgrounds()
            chromaticCorrector = optimizeTask.get_chromatic_corrector()
        else:
            codebook = self.preprocessTask.get_codebook()
            scaleFactors = np.ones(self.get_codebook().get_bit_count())
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
            chromaticCorrector = aberration.RigidChromaticCorrector(
                self._get_initial_chromatic_corrector(), 
                self._get_reference_color())

        zPositionCount = len(self.dataSet.get_z_positions())
        channelCount = self.get_codebook().get_bit_count() + \
                        len(self.parameters['feature_channels'])

        # avoids creating a huge numpy matrix such as 100 x 22 x 2048 x 2048
        # by proecessing every z indepedently. This is mostly for 3D-MERFISH
        # when each fov contains 100s of z slices
        if self.parameters['file_type'] == "dax":
            outputTif = imagewriter.DaxWriter(
                self.dataSet._analysis_image_name(
                    self, "image_", fragmentIndex).replace('tif', 'dax')
                )
        else:
            outputTif = imagewriter.TiffWriter(
                self.dataSet._analysis_image_name(
                    self, "image_", fragmentIndex)
                )
            
        
        for zIndex in range(zPositionCount):
            # chromaticCorrector is None if optimization task is not given
            imageSet_preproc = self.preprocessTask.get_processed_image_set(
                fragmentIndex, zIndex, chromaticCorrector)

            # apply low pass filter
            imageSet_preproc = np.array([ 
                imagefilters.low_pass_filter(
                    imageSet_preproc[i,:,:],
                    self.parameters['lowpass_sigma']) \
                        for i in range(imageSet_preproc.shape[0]) ]
                        ).astype(np.uint16)

            # return None if feature_channels is empty - [] 
            imageSet_feature = self.get_feature_image_set(
                fragmentIndex, zIndex, 
                self.parameters['feature_channels'], 
                chromaticCorrector)
            
            # combine feature set and processed image set
            if imageSet_feature is not None:
                imageSet_feature = imageSet_feature.astype(np.uint16)
                imageSet = np.concatenate([
                    imageSet_preproc, imageSet_feature], axis=0)
            else:
                imageSet = imageSet_preproc
            
            for i in range(imageSet.shape[0]):
                    outputTif.addFrame(imageSet[i])
        
        outputTif.close()


        