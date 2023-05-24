import numpy as np
import pandas
import os
import tempfile
from skimage import transform
from typing import Dict
from typing import List

from merlin.util import aberration
from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.data.codebook import Codebook
from merlin.util import barcodefilters

class BarcodeSavingParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that barcodes barcodes into a barcode database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_barcode_database().empty_database(fragmentIndex)

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """ Get the barcode database this analysis task saves barcodes into.

        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)


class Decode(BarcodeSavingParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    def __init__(self, dataSet: dataset.MERFISHDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'write_processed_images' not in self.parameters:
            self.parameters['write_processed_images'] = False
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 2
        if 'magnitude_threshold' not in self.parameters:
            self.parameters['magnitude_threshold'] = 10
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.65
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decode_3d' not in self.parameters:
            self.parameters['decode_3d'] = False
        if 'memory_map' not in self.parameters:
            self.parameters['memory_map'] = False
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_zPlane_threshold' not in self.parameters:
                self.parameters['z_duplicate_zPlane_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]

        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()
    
    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})
    
    def _get_reference_color(self):
        return min(self._get_used_colors())

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        bitCount = self.get_codebook().get_bit_count()

        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = preprocessTask.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
            # Add two to match matlab code.
            # TODO: Does +2 make sense? Used to be consistent with Matlab code
            initialScaleFactors[i] = \
                np.argmin(np.abs(cumulativeHistogram-0.9)) + 2

        return initialScaleFactors

    def _get_initial_chromatic_corrector(self):
        usedColors = self._get_used_colors()
        return {u: {v: transform.SimilarityTransform()
            for v in usedColors if v >= u} for u in usedColors}

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        
        decode3d = self.parameters['decode_3d']
        if 'optimize_task' in self.parameters:
            optimizeTask = self.dataSet.load_analysis_task(
            		self.parameters['optimize_task'])
            scaleFactors = optimizeTask.get_scale_factors() 
            backgrounds = optimizeTask.get_backgrounds()
            chromaticCorrector = optimizeTask.get_chromatic_corrector()
        else:
            codebook = self.get_codebook()
            scaleFactors = np.ones(self.get_codebook().get_bit_count())
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
            chromaticCorrector = aberration.RigidChromaticCorrector(
                self._get_initial_chromatic_corrector(), 
                self._get_reference_color())
 
        lowPassSigma = self.parameters['lowpass_sigma']

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)

        zPositionCount = len(self.dataSet.get_z_positions())
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), 
                                dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape),
                                  dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), 
                            dtype=np.float32)

        if self.parameters['write_processed_images']:
            processedImages = np.array([ preprocessTask.get_processed_image_set(
               fragmentIndex, zIndex, chromaticCorrector) \
                   for zIndex in range(zPositionCount) ])
            self._save_processed_images(
                fragmentIndex, zPositionCount, processedImages)
            del processedImages
        
        if not decode3d:
            for zIndex in range(zPositionCount):
                di, pm, d = self._process_independent_z_slice(
                    fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                    backgrounds, preprocessTask, decoder)

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

        else:
            with tempfile.TemporaryDirectory() as tempDirectory:
                if self.parameters['memory_map']:
                    normalizedPixelTraces = np.memmap(
                        os.path.join(tempDirectory, 'pixel_traces.dat'),
                        mode='w+', dtype=np.float32,
                        shape=(zPositionCount, bitCount, *imageShape))
                else:
                    normalizedPixelTraces = np.zeros(
                        (zPositionCount, bitCount, *imageShape),
                        dtype=np.float32)

                for zIndex in range(zPositionCount):
                    imageSet = preprocessTask.get_processed_image_set(
                        fragmentIndex, zIndex, chromaticCorrector)
                    imageSet = imageSet.reshape(
                        (imageSet.shape[0], imageSet.shape[-2],
                         imageSet.shape[-1]))

                    di, pm, npt, d = decoder.decode_pixels(
                        imageSet, scaleFactors, backgrounds,
                        lowPassSigma=lowPassSigma,
                        distanceThreshold=self.parameters['distance_threshold'],
                        magnitudeThreshold=self.parameters['magnitude_threshold'])

                    normalizedPixelTraces[zIndex, :, :, :] = npt
                    decodedImages[zIndex, :, :] = di
                    magnitudeImages[zIndex, :, :] = pm
                    distances[zIndex, :, :] = d

                self._extract_and_save_barcodes(
                    decoder, decodedImages, magnitudeImages,
                    normalizedPixelTraces,
                    distances, fragmentIndex)

                del normalizedPixelTraces

        if self.parameters['write_decoded_images']:
            imageSize = decodedImages.shape
            self._save_decoded_images(
                fragmentIndex, 1, 
				decodedImages[0].reshape([1,imageSize[1], imageSize[2]]), 
                magnitudeImages[0].reshape([1, imageSize[1], imageSize[2]]),
                distances[0].reshape([1, imageSize[1], imageSize[2]]))

        if self.parameters['remove_z_duplicated_barcodes']:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(
                bcDB.get_barcodes(fov=fragmentIndex))
            bcDB.empty_database(fragmentIndex)
            bcDB.write_barcodes(bc, fov=fragmentIndex)


    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder):

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

        di, pm, npt, d = decoder.decode_pixels(
            imageSet, scaleFactors, backgrounds,
            lowPassSigma=self.parameters['lowpass_sigma'],
            distanceThreshold=self.parameters['distance_threshold'],
            magnitudeThreshold=self.parameters['magnitude_threshold'])

        self._extract_and_save_barcodes(
            decoder, di, pm, npt, d, fov, zIndex)

        return di, pm, d

    def _save_processed_images(self, fov: int, zPositionCount: int,
                             processedImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, processedImages.shape[1])
            with self.dataSet.writer_for_analysis_images(
                    self, 'processed', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(processedImages[i].astype(np.uint16),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, 3)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(magnitudeImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(distanceImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zIndex: int=None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        minimumArea = self.parameters['minimum_area']
        
        barcodes = pandas.concat([decoder.extract_barcodes_with_index(
                i, decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
                self.cropWidth, zIndex, globalTask, minimumArea)
                for i in range(self.get_codebook().get_barcode_count())])
        
        z_pos_list = np.array(self.dataSet.get_z_positions())
        barcodes = barcodes.assign(global_z =
            [ z_pos_list[x] for x in barcodes.z.astype(int) ])
        
        self.get_barcode_database().write_barcodes(barcodes, fov=fov)

    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_zPlane_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc


class DecodeML(BarcodeSavingParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images using machine learning.
    """

    def __init__(self, dataSet: dataset.MERFISHDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'write_processed_images' not in self.parameters:
            self.parameters['write_processed_images'] = False
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'minimum_prob' not in self.parameters:
            self.parameters['minimum_prob'] = 0
        if 'magnitude_threshold' not in self.parameters:
            self.parameters['magnitude_threshold'] = 10
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.65
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decode_3d' not in self.parameters:
            self.parameters['decode_3d'] = False
        if 'memory_map' not in self.parameters:
            self.parameters['memory_map'] = False
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_zPlane_threshold' not in self.parameters:
                self.parameters['z_duplicate_zPlane_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)

        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]
        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})
    
    def _get_reference_color(self):
        return min(self._get_used_colors())

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        bitCount = self.get_codebook().get_bit_count()

        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = preprocessTask.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
            # Add two to match matlab code.
            # TODO: Does +2 make sense? Used to be consistent with Matlab code
            initialScaleFactors[i] = \
                np.argmin(np.abs(cumulativeHistogram-0.9)) + 2

        return initialScaleFactors

    def _get_initial_chromatic_corrector(self):
        usedColors = self._get_used_colors()
        return {u: {v: transform.SimilarityTransform()
            for v in usedColors if v >= u} for u in usedColors}

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        
        decode3d = self.parameters['decode_3d']
        optimizeTask = self.dataSet.load_analysis_task(
        		self.parameters['optimize_task'])
        scaleFactors = optimizeTask.get_scale_factors()
        backgrounds = optimizeTask.get_backgrounds()
        chromaticCorrector = optimizeTask.get_chromatic_corrector()
        pixelScoreMachine = optimizeTask.get_pixel_score_machine()
 
        lowPassSigma = self.parameters['lowpass_sigma']

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)

        zPositionCount = len(self.dataSet.get_z_positions())
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape),
                                   dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), dtype=np.float32)
        probImages = np.zeros((zPositionCount, *imageShape), dtype=np.float32)

        if self.parameters['write_processed_images']:
            processedImages = np.array([ preprocessTask.get_processed_image_set(
               fragmentIndex, zIndex, chromaticCorrector) \
                   for zIndex in range(zPositionCount) ])
            self._save_processed_images(
                fragmentIndex, zPositionCount, processedImages)
            del processedImages
        
        if not decode3d:
            for zIndex in range(zPositionCount):
                di, pm, d, p = self._process_independent_z_slice(
                    fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                    backgrounds, preprocessTask, decoder, pixelScoreMachine
                )

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d
                probImages[zIndex, :, :] = p

        else:
            with tempfile.TemporaryDirectory() as tempDirectory:
                if self.parameters['memory_map']:
                    normalizedPixelTraces = np.memmap(
                        os.path.join(tempDirectory, 'pixel_traces.dat'),
                        mode='w+', dtype=np.float32,
                        shape=(zPositionCount, bitCount, *imageShape))
                else:
                    normalizedPixelTraces = np.zeros(
                        (zPositionCount, bitCount, *imageShape),
                        dtype=np.float32)

                for zIndex in range(zPositionCount):
                    imageSet = preprocessTask.get_processed_image_set(
                        fragmentIndex, zIndex, chromaticCorrector)
                    imageSet = imageSet.reshape(
                        (imageSet.shape[0], imageSet.shape[-2],
                         imageSet.shape[-1]))

                    di, pm, npt, d, p = decoder.decode_pixels_ml(
                        imageSet, scaleFactors, backgrounds,
                        lowPassSigma=lowPassSigma,
                        pixelScoreMachine=pixelScoreMachine,
                        distanceThreshold=self.parameters['distance_threshold'],
                        magnitudeThreshold=self.parameters['magnitude_threshold']
                        )

                    normalizedPixelTraces[zIndex, :, :, :] = npt
                    decodedImages[zIndex, :, :] = di
                    magnitudeImages[zIndex, :, :] = pm
                    distances[zIndex, :, :] = d
                    probImages[zIndex, :, :] = p
                
                self._extract_and_save_barcodes(
                    decoder, decodedImages, magnitudeImages,
                    normalizedPixelTraces,
                    distances, probImages, fragmentIndex)

                del normalizedPixelTraces

        if self.parameters['write_decoded_images']:
            imageSize = decodedImages.shape
            self._save_decoded_images(
                fragmentIndex, 1, 
				decodedImages[0].reshape([1,imageSize[1], imageSize[2]]), 
                magnitudeImages[0].reshape([1, imageSize[1], imageSize[2]]),
                distances[0].reshape([1, imageSize[1], imageSize[2]]),
                probImages[0].reshape([1, imageSize[1], imageSize[2]]))

        if self.parameters['remove_z_duplicated_barcodes']:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(
                bcDB.get_barcodes(fov=fragmentIndex))
            bcDB.empty_database(fragmentIndex)
            bcDB.write_barcodes(bc, fov=fragmentIndex)

    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder, pixelScoreMachine):

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

        di, pm, npt, d, p = decoder.decode_pixels_ml(
            imageSet, scaleFactors, backgrounds, pixelScoreMachine,
            lowPassSigma=self.parameters['lowpass_sigma'],
            distanceThreshold=self.parameters['distance_threshold'],
            magnitudeThreshold=self.parameters['magnitude_threshold']
            )
        self._extract_and_save_barcodes(
            decoder, di, pm, npt, d, p, fov, zIndex)

        return di, pm, d, p

    def _save_processed_images(self, fov: int, zPositionCount: int,
                             processedImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, processedImages.shape[1])
            with self.dataSet.writer_for_analysis_images(
                    self, 'processed', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(processedImages[i].astype(np.uint16),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray,
                             probImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, 4)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(magnitudeImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(distanceImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)
                    outputTif.save(probImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, pixelProbs: np.ndarray, 
            fov: int, zIndex: int=None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        minimumArea = self.parameters['minimum_area']
        minimumProb = self.parameters['minimum_prob']
        
        barcodes = pandas.concat([decoder.extract_barcodes_with_index_ml(
                i, decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
                self.cropWidth, zIndex, globalTask, minimumArea)
                for i in range(self.get_codebook().get_barcode_count())])

        z_pos_list = np.array(self.dataSet.get_z_positions())
        barcodes = barcodes.assign(global_z =
            [ z_pos_list[x] for x in barcodes.z.astype(int) ])

        self.get_barcode_database().write_barcodes(barcodes, fov=fov)

    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_zPlane_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc

