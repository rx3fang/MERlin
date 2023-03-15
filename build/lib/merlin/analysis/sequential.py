import pandas
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops
from functools import reduce

from merlin.core import analysistask
from merlin.util import imagefilters

"""
A module for sequential round analysis
Latest Update: Rongxin Fang 11/12/2022
"""

class SumSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.highpass = False
        if 'highpass_sigma' in self.parameters:
            self.highpass = True
        
        if 'z_indexes' not in self.parameters:
            zPositionCount = len(self.dataSet.get_z_positions())
            self.parameters['z_indexes'] = range(zPositionCount)

        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _extract_signal(self, cells, inputImage, zIndex) -> pandas.DataFrame:
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zIndex]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions:
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                    pixels.append(np.array(
                                self.alignTask.global_coordinates_to_fov(
                                    cell.get_fov(), xyZip)))
                cellCoords.append(pixels)

        cellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))]
        mask = np.zeros(inputImage.shape, np.uint8)
        for i, cell in enumerate(cellCoords):
            cv2.drawContours(mask, cell, -1, i+1, -1)
        propsDict = {x.label: x for x in regionprops(mask, inputImage)}
        propsOut = pandas.DataFrame(
            data=[(propsDict[k].intensity_image.sum(),
                   propsDict[k].filled_area)
                  if k in propsDict else (0, 0)
                  for k in range(1, len(cellCoords) + 1)],
            index=cellIDs,
            columns=['Intensity', 'Pixels'])
        return propsOut

    def _get_sum_signal(self, fov, channels, zIndex):

        warpTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        segmentTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        cells = segmentTask.get_feature_database().read_features(fov)

        signals = []
        for ch in channels:
            img = warpTask.get_aligned_image(fov, ch, zIndex)
            if self.highpass:
                highPassSigma = self.parameters['highpass_sigma']
                highPassFilterSize = int(2 * np.ceil(3 * highPassSigma) + 1)
                img = imagefilters.high_pass_filter(img,
                                                    highPassFilterSize,
                                                    highPassSigma)
            signals.append(self._extract_signal(cells, img,
                                                zIndex).iloc[:, [0]])

        # adding num of pixels
        signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [1]])

        compiledSignal = pandas.concat(signals, 1)
        compiledSignal.columns = channels+['Pixels']
        
        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fov, 'signals', index_col=0)

    def _run_analysis(self, fragmentIndex):
        # load input parameters
        zIndexes = self.parameters['z_indexes']
        channelNames = self.parameters['channel_names']

        # get channel ids
        channels = [ self.dataSet.get_data_organization().\
            get_data_channel_with_name(cn) \
            for cn in channelNames ]
        
        # get sum signal for each z plane
        sumSignalList = [ self._get_sum_signal(
                fragmentIndex, channels, zIndex) \
            for zIndex in zIndexes ]
        
        # sum signals over all z planes
        sumSignal = reduce(lambda x, y: x.add(y, fill_value=0), 
            sumSignalList)
        
        # normalize signal by pixel number
        normSignal = sumSignal.iloc[:, :-1].\
            div(sumSignal.loc[:, 'Pixels'], 0)
        
        # rename the column name
        normSignal.columns = channelNames
        
        # save as csv file
        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal', 
                self.get_analysis_name(),
                fragmentIndex, 'signals')

class ExportSumSignals(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['sequential_task']]

    def _run_analysis(self):
        sTask = self.dataSet.load_analysis_task(
                    self.parameters['sequential_task'])
        signals = sTask.get_sum_signals()

        self.dataSet.save_dataframe_to_csv(
                    signals, 'sequential_sum_signals',
                    self.get_analysis_name())
