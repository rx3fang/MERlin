import os
import cv2
import numpy as np

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
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

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
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
            self.parameters['save_pixel_histogram'] = True

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
    
    ## TODO: add a flat field image to remove the background
    # this is particularly important for 650 channel of W1 spinning disk
    # Background remove method should include "subtract" (default) and "divide"
    
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
        filteredImage = (inputImage - imageDark) / imageFlat
        
        # high pass filter to remove background
        filteredImage = self._high_pass_filter(filteredImage)
        
        # deconvolution (disabled when _deconSigma is -1)
        deconFilterSize = self.parameters['decon_filter_size']
        if self._deconSigma == -1:
            deconvolvedImage = filteredImage.astype(np.uint16)
        else:
            deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
                filteredImage, deconFilterSize, self._deconSigma,
                self._deconIterations).astype(np.uint16)
        return deconvolvedImage

    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                self._highPassSigma)
        return hpImage.astype(np.float)
    
    def _run_analysis(self, fragmentIndex):
        
        if not self.parameters['save_pixel_histogram']:
            pass

        warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
        
        histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
        pixelHistogram = np.zeros(
                (self.get_codebook().get_bit_count(), len(histogramBins)-1))
        
        for bi, b in enumerate(self.get_codebook().get_bit_names()):
            dataChannel = self.dataSet.get_data_organization()\
                    .get_data_channel_for_bit(b)
            for i in range(len(self.dataSet.get_z_positions())):
                inputImage = warpTask.get_aligned_image(
                        fragmentIndex, dataChannel, i)
                
                imageColor = self.dataSet.get_data_organization()\
                                .get_data_channel_color(dataChannel)
        
                deconvolvedImage = self._preprocess_image(
                    inputImage, imageColor)
        
                pixelHistogram[bi, :] += np.histogram(
                        deconvolvedImage, bins=histogramBins)[0]
        
        self._save_pixel_histogram(pixelHistogram, fragmentIndex)

class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        if 'decon_iterations' not in parameters:
            self.parameters['decon_iterations'] = 2

        self._deconIterations = self.parameters['decon_iterations']

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson_guo(
            filteredImage, deconFilterSize, self._deconSigma,
            self._deconIterations).astype(np.uint16)
        return deconvolvedImage
