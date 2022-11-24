import os
import tifffile
import subprocess
import random
import numpy as np
import pandas as pd
from merlin import dataset

from merlin.core import analysistask
from merlin.util import barcodedb

"""
A module for thunderstorm fitting
This could actually be used for compressing the data
Rongxin Fang 11/11/22
"""

class ThunderstormSavingParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that barcodes barcodes into a barcode database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """ Get the barcode database this analysis task saves barcodes into.
        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)

class Thunderstorm(ThunderstormSavingParallelAnalysisTask):

    """
    An analysis tast that runs thunderSTORMS to identify molecule locolization
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 1
        
        if 'crop_width' not in self.parameters:
            self.crop_width = 101
        else:
            self.crop_width = self.parameters['crop_width']
            
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        
        self.imageSize = dataSet.get_image_dimensions()
        
    def get_codebook(self):
        return self.dataSet.get_codebook(
            self.parameters['codebook_index'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        dependencies = [self.parameters['global_align_task'],
                        self.parameters['warp_task']]
        return dependencies
    
    def _analysis_file_name(self,
                            analysisTask,
                            subdirectory: str,
                            fileBaseName: str, 
                            fovIndex: int,
                            fileType = "tif"
                            ) -> str:
        
        destPath = self.dataSet.get_analysis_subdirectory(
                analysisTask, subdirectory=subdirectory)
                
        return os.sep.join([destPath, 
            fileBaseName+"_"+str(fovIndex)+'.'+fileType])
    
    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector = None
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
            chromaticCorrector = None
    ) -> np.ndarray:
        return self.warpTask.get_aligned_image(
            fov, dataChannel, zIndex,
            chromaticCorrector)
    
    def _run_analysis(self, fragmentIndex):
        
        codebook = self.get_codebook()
        bitCount = codebook.get_bit_count()
        zPositionCount = len(self.dataSet.get_z_positions())
        
        imageShape = self.dataSet.get_image_dimensions()
        imageSet = np.zeros((zPositionCount, bitCount, *imageShape), dtype=np.int16)
        
        pixelSize = self.dataSet.get_microns_per_pixel() * 1000
        
        imageName = self._analysis_file_name(
                self,
                "images", "fov",
                fragmentIndex, 
                "tif")
        
        csvName = self._analysis_file_name(
                self,
                "csv", "fov",
                fragmentIndex, 
                "csv")
        
        macroName = self._analysis_file_name(
                self,
                "macros", "fov",
                fragmentIndex, 
                "ijm")

        for zIndex in range(zPositionCount):
            images = self.get_processed_image_set(fragmentIndex, zIndex)        
            imageSet[zIndex, :, :, :] = images
        
        tifffile.imwrite(data=imageSet.astype(np.uint16),
                         file=imageName)
        
        # create macro ijm file
        macroLines = """open("%s");
            run("Camera setup", "offset=414.0 isemgain=false photons2adu=3.6 pixelsize=%0.1f");
            run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.5 fitradius=5 method=[Maximum likelihood] full_image_fitting=false mfaenabled=false renderer=[Averaged shifted histograms] magnification=8.0 colorizez=false threed=false shifts=3 repaint=50");
            run("Export results", "filepath=%s fileformat=[CSV (comma separated)] sigma=true intensity=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
            run("Quit")""" % (imageName, pixelSize, csvName)
        
        # write down the macro file
        with open(macroName, "w") as fout:
                fout.write(macroLines)
        
        # run thunderSTROM
        ijpath = self.parameters['ij_path']
        bashCommand = "xvfb-run -a java -jar {ijpath}/ij.jar ".format(ijpath=ijpath) + \
            " --console --headless -ijpath {ijpath} ".format(ijpath=ijpath) + \
            " -macro {macroFile}".format(macroFile = macroName)
        
        try:
            subprocess.check_call(bashCommand, shell=True)
        except subprocess.CalledProcessError:
            raise Exception('Unable to finsh thunderSTORM.')

        barcodes = pd.read_csv(csvName)
        barcodes.frame = barcodes.frame - 1
        
        barcodeId = dict(zip(list(range(bitCount * zPositionCount)), 
            list(range(bitCount)) * zPositionCount))
        
        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        barcodeZindex = dict(zip(list(range(bitCount * zPositionCount)), 
            list(np.repeat(np.arange(zPositionCount),bitCount))))
        
        barcodeZpos = dict(zip(list(range(bitCount * zPositionCount)),
            list(np.repeat(zPos,bitCount))))

        barcodes = barcodes.assign(
            barcode_id = [ barcodeId[i] for i in barcodes.frame ])

        barcodes = barcodes.assign(x = barcodes["x [nm]"] / pixelSize)
        barcodes = barcodes.assign(y = barcodes["y [nm]"] / pixelSize)
        global_x = []
        global_y = []
        for i, x in barcodes.iterrows():
            (gx, gy) = self.alignTask.fov_coordinates_to_global(
                fragmentIndex, [x.x, x.y])
            global_x.append(gx)
            global_y.append(gy)
        
        barcodes = barcodes.assign(global_x = global_x)
        barcodes = barcodes.assign(global_y = global_y)
        
        barcodes = barcodes.assign(
            global_z = [ barcodeZpos[i] for i in barcodes.frame ])
        barcodes = barcodes.assign(
            z = [ barcodeZindex[i] for i in barcodes.frame ])

        # this is required for partioning barcodes.
        barcodes = barcodes.assign(fov = fragmentIndex)
        barcodes = barcodes.assign(mean_intensity = barcodes["intensity [photon]"])
        barcodes = barcodes.assign(max_intensity = barcodes["intensity [photon]"])
        barcodes = barcodes.assign(area = 1)
        barcodes = barcodes.assign(mean_distance = 1)
        barcodes = barcodes.assign(min_distance = 1)
        barcodes = barcodes.assign(cell_index = str(-1))

        # remove barcodes within the edge
        barcodes = barcodes[barcodes.x >= self.crop_width]
        barcodes = barcodes[barcodes.x <= (self.imageSize[0] - self.crop_width)]
        barcodes = barcodes[barcodes.y >= self.crop_width]
        barcodes = barcodes[barcodes.y <= (self.imageSize[1] - self.crop_width)]

        for i in range(bitCount):
            barcodes['intensity_'+str(i)] = 0

        self.get_barcode_database().write_barcodes(
            barcodes, fov=fragmentIndex)

        if not self.parameters['keep_images']:
            os.remove(imageName)

class ThunderstormEstimateThreshold(analysistask.AnalysisTask):

    """
    An analysis task that determines the intensity threshold for 
    thunderSTORM identified locolizations
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'fov_num' not in self.parameters:
            self.parameters['fov_num'] = 50

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['thunderstorm_task']]
    
    def get_thresholds(self):
        return self.dataSet.load_pickle_analysis_result(
                "intensityThreshold", self.analysisName)
    
    def _run_analysis(self):

        # load the thunderSTORM task
        thunderstormTask = self.dataSet.load_analysis_task(
                self.parameters['thunderstorm_task'])

        codebook = thunderstormTask.get_codebook()
        bitCount = codebook.get_bit_count()

        # all the fovs
        fovList = self.dataSet.get_fovs()    
        
        # randomly sample 50 fov
        fovListSub = random.sample(list(fovList), 
            min(self.parameters['fov_num'], len(fovList)))
        
        # load all the barcodes
        barcodeDB = thunderstormTask.get_barcode_database()
        barcodes = pd.concat([ barcodeDB.get_barcodes(fov=fragmentIndex) \
            for fragmentIndex in fovList ], axis=0)
        
        thresholdDict = {}
        for barcode_id in range(bitCount):
            barcodesSub = barcodes[barcodes.barcode_id == barcode_id]

            # estimate the threshold
            intensityVec = np.log10(barcodesSub["mean_intensity"] + 1)
            intensityThreshold = np.mean(intensityVec) + np.std(intensityVec)
            thresholdDict[barcode_id] = 10 ** intensityThreshold
        
        self.dataSet.save_pickle_analysis_result(
                thresholdDict, "intensityThreshold", self.analysisName)

class ThunderstormAdaptiveFilter(ThunderstormSavingParallelAnalysisTask):

    """
    An analysis task to filter locolizations based on the threshold estimated
    by EstimateThreshold
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())
    
    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1
    
    def get_dependencies(self):
        dependencies = [self.parameters['thunderstorm_estimate_threshold_task'],
                        self.parameters['thunderstorm_task']]
        return dependencies

    def get_codebook(self):
        thunderstormTask = self.dataSet.load_analysis_task(
                self.parameters['thunderstorm_task'])
        return thunderstormTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        # load estimate threshold task
        thresholdTask = self.dataSet.load_analysis_task(
            self.parameters['thunderstorm_estimate_threshold_task'])

        # load thunderstorm task
        thunderstormTask = self.dataSet.load_analysis_task(
            self.parameters['thunderstorm_task'])
        
        codebook = thunderstormTask.get_codebook()
        bitCount = codebook.get_bit_count()
        thresholds = thresholdTask.get_thresholds()
        
        barcodes = thunderstormTask.get_barcode_database()\
            .get_barcodes(fov=fragmentIndex)
        
        barcodesFiltered = pd.concat([ 
                barcodes[(barcodes.barcode_id == bit) & \
                    (barcodes.mean_intensity > thresholds[bit])] \
            for bit in range(bitCount) ], axis=0)
        
        self.get_barcode_database().write_barcodes(
            barcodeInformation = barcodesFiltered, 
            fov=fragmentIndex)

class ThunderstormFilter(ThunderstormSavingParallelAnalysisTask):

    """
    An analysis task to filter locolizations based on user defined
    thresholds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())
    
    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1
    
    def get_dependencies(self):
        dependencies = [self.parameters['thunderstorm_task']]
        return dependencies

    def get_codebook(self):
        thunderstormTask = self.dataSet.load_analysis_task(
                self.parameters['thunderstorm_task'])
        return thunderstormTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        
        if 'min_intensity' not in self.parameters:
            self.parameters['min_intensity'] = 0

        if 'max_intensity' not in self.parameters:
            self.parameters['max_intensity'] = float('inf')

        if 'min_offset' not in self.parameters:
            self.parameters['min_offset'] = 0

        if 'max_offset' not in self.parameters:
            self.parameters['max_offset'] = float('inf')

        if 'min_bkgstd' not in self.parameters:
            self.parameters['min_bkgstd'] = 0

        if 'max_bkgstd' not in self.parameters:
            self.parameters['max_bkgstd'] = float('inf')

        if 'min_uncertainty' not in self.parameters:
            self.parameters['min_uncertainty'] = 0

        if 'max_uncertainty' not in self.parameters:
            self.parameters['max_uncertainty'] = float('inf')

        if 'min_sigma' not in self.parameters:
            self.parameters['min_sigma'] = 0

        if 'max_sigma' not in self.parameters:
            self.parameters['max_sigma'] = float('inf')

        # load thunderstorm task
        thunderstormTask = self.dataSet.load_analysis_task(
            self.parameters['thunderstorm_task'])
        
        codebook = thunderstormTask.get_codebook()
        bitCount = codebook.get_bit_count()
        thresholds = thresholdTask.get_thresholds()
        
        barcodes = thunderstormTask.get_barcode_database()\
            .get_barcodes(fov=fragmentIndex)
        
        barcodesFiltered = barcodes[
            ( barcodes["sigma [nm]"] > self.parameters['min_sigma'] ) & ( barcodes["sigma [nm]"] < self.parameters['max_sigma'] ) & \
            ( barcodes["intensity [photon]"] > self.parameters['min_intensity'] ) & ( barcodes["intensity [photon]"] < self.parameters['max_intensity'] ) & \
            ( barcodes["offset [photon]"] > self.parameters['min_offset'] ) & ( barcodes["offset [photon]"] < self.parameters['max_offset'] ) & \
            ( barcodes["bkgstd [photon]"] > self.parameters['min_bkgstd'] ) & ( barcodes["bkgstd [photon]"] < self.parameters['max_bkgstd'] ) & \
            ( barcodes["uncertainty [nm]"] > self.parameters['min_uncertainty'] ) & ( barcodes["uncertainty [nm]"] < self.parameters['max_uncertainty'] ) ]

        self.get_barcode_database().write_barcodes(
            barcodesFiltered, fov=fragmentIndex)



