import numpy as np
import pandas
from scipy import optimize

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from merlin.core import analysistask
from merlin.analysis import decode

class AbstractFilterBarcodes(decode.BarcodeSavingParallelAnalysisTask):
    """
    An abstract class for filtering barcodes identified by pixel-based decoding.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_codebook(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        return decodeTask.get_codebook()

class FilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 1e6

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        distanceThreshold = self.parameters['distance_threshold']
        barcodeDB = self.get_barcode_database()
        barcodeDB.write_barcodes(
            decodeTask.get_barcode_database().get_filtered_barcodes(
                areaThreshold, intensityThreshold,
                distanceThreshold=distanceThreshold, fov=fragmentIndex),
            fov=fragmentIndex)


class EstimateLikelihoodThreshold(analysistask.AnalysisTask):

    """
    An analysis task that estimate the loglikelihood threshold
    for barcodes 
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'bins' not in self.parameters:
            self.parameters['bins'] = 5000
        if 'fov_num' not in self.parameters:
            self.parameters['fov_num'] = 50
        if 'random_seed' not in self.parameters:
            self.parameters['random_seed'] = 1
            
        if 'fov_index' in self.parameters:
            logger = self.dataSet.get_logger(self)
            logger.info('Setting fov_per_iteration to length of fov_index')

            self.parameters['fov_num'] = \
                len(self.parameters['fov_index'])
        else:
            self.parameters['fov_index'] = []
            if self.parameters['random_seed'] != -1:
                np.random.seed(self.parameters['random_seed'])

            for i in range(self.parameters['fov_num']):
                fovIndex = int(np.random.choice(
                    list(self.dataSet.get_fovs())))
                self.parameters['fov_index'].append(fovIndex)
        
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 1800

    def get_dependencies(self):
        return [self.parameters['run_after_task']]
    
    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()
        
        barcodes = pandas.concat([ barcodeDB.get_barcodes(fov=fragmentIndex) \
            for fragmentIndex in self.parameters['fov_index'] ], axis=0)

        barcodes = pandas.concat([ barcodeDB.get_barcodes(fov=fragmentIndex) \
            for fragmentIndex in range(20)], axis=0)
        
        misidentificationRates = self.estimate_lik_err_table(
            barcodes, codebook, 
            minScore=barcodes.loglikehood.min(),
            maxScore=barcodes.loglikehood.max(),
            bins = self.parameters['bins'])
        
        self.dataSet.save_pickle_analysis_result(
            misidentificationRates, 'misidentification_rates',
            self.analysisName)
     
    def calculate_threshold_for_misidentification_rate(
            self, targetMisidentificationRate: float) -> float:
        
        misidentificationRates = self.dataSet.load_pickle_analysis_result(
            'misidentification_rates', self.analysisName)
        
        return min(np.array(list(misidentificationRates.keys()))[
                np.array(list(misidentificationRates.values())) <= \
                    targetMisidentificationRate])
    
    def extract_barcodes_with_threshold(self, blankThreshold: float,
                                        barcodeSet: pandas.DataFrame
                                        ) -> pandas.DataFrame:
        return barcodeSet[barcodeSet.loglikehood >= blankThreshold]
    
    
    @staticmethod
    def estimate_lik_err_table(
        bd, cb, minScore=0, maxScore=10, bins=1000):
        
        scores = np.linspace(minScore, maxScore, bins)
        blnkBarcodeNum = len(cb.get_blank_indexes())
        codeBarcodeNum = len(cb.get_coding_indexes()) + len(cb.get_blank_indexes())
        pvalues = dict()
        for s in scores:
            bd = bd[bd.loglikehood >= s]
            numPos = np.count_nonzero(
                bd.barcode_id.isin(cb.get_coding_indexes()))
            numNeg = np.count_nonzero(
                bd.barcode_id.isin(cb.get_blank_indexes()))
            numNegPerBarcode = numNeg / blnkBarcodeNum
            numPosPerBarcode = (numPos + numNeg) / codeBarcodeNum
            pvalues[s] = numNegPerBarcode / numPosPerBarcode
        return pvalues

class FilterBarcodesLikelihood(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on the likelihood of each
    barcodes estimated based on barcode intensity distance and area.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)

        bcDatabase.write_barcodes(adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes), fov=fragmentIndex)

class GenerateAdaptiveThreshold(analysistask.AnalysisTask):

    """
    An analysis task that generates a three-dimension mean intenisty,
    area, minimum distance histogram for barcodes as they are decoded.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'tolerance' not in self.parameters:
            self.parameters['tolerance'] = 0.001
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 1800

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def get_blank_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('blank_counts', self)

    def get_coding_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('coding_counts', self)

    def get_total_count_histogram(self) -> np.ndarray:
        return self.get_blank_count_histogram() \
               + self.get_coding_count_histogram()

    def get_area_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('area_bins', self)

    def get_distance_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'distance_bins', self)

    def get_intensity_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'intensity_bins', self, None)

    def get_blank_fraction_histogram(self) -> np.ndarray:
        """ Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram()
        totalHistogram = self.get_coding_count_histogram()
        blankFraction = blankHistogram / totalHistogram
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankFraction /= blankBarcodeCount/(
                blankBarcodeCount + codingBarcodeCount)
        return blankFraction

    def calculate_misidentification_rate_for_threshold(
            self, threshold: float) -> float:
        """ Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return ((blankCounts/blankBarcodeCount) /
                (codingCounts/codingBarcodeCount))

    def calculate_threshold_for_misidentification_rate(
            self, targetMisidentificationRate: float) -> float:
        """ Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        tolerance = self.parameters['tolerance']
        def misidentification_rate_error_for_threshold(x, targetError):
            return self.calculate_misidentification_rate_for_threshold(x) \
                - targetError
        return optimize.newton(
            misidentification_rate_error_for_threshold, 0.2,
            args=[targetMisidentificationRate], tol=tolerance, x1=0.3,
            disp=False)

    def calculate_barcode_count_for_threshold(self, threshold: float) -> float:
        """ Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()
        return np.sum(blankHistogram[blankFraction < threshold]) \
            + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, blankThreshold: float,
                                        barcodeSet: pandas.DataFrame
                                        ) -> pandas.DataFrame:
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        selectData[:, 0] = np.log10(selectData[:, 0])
        blankFractionHistogram = self.get_blank_fraction_histogram()

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], self.get_intensity_bins(),
                         right=True),
             np.digitize(selectData[:, 1], self.get_distance_bins(),
                         right=True),
             np.digitize(selectData[:, 2], self.get_area_bins()))) - 1
        barcodeBins[0, :] = np.clip(
            barcodeBins[0, :], 0, blankFractionHistogram.shape[0]-1)
        barcodeBins[1, :] = np.clip(
            barcodeBins[1, :], 0, blankFractionHistogram.shape[1]-1)
        barcodeBins[2, :] = np.clip(
            barcodeBins[2, :], 0, blankFractionHistogram.shape[2]-1)
        raveledIndexes = np.ravel_multi_index(
            barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    @staticmethod
    def _extract_counts(barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        barcodeData[:, 0] = np.log10(barcodeData[:, 0])
        return np.histogramdd(
            barcodeData, bins=(intensityBins, distanceBins, areaBins))[0]

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()

        completeFragments = \
            self.dataSet.load_numpy_analysis_result_if_available(
                'complete_fragments', self, [False]*self.fragment_count())
        pendingFragments = [
            decodeTask.is_complete(i) and not completeFragments[i]
            for i in range(self.fragment_count())]

        areaBins = self.dataSet.load_numpy_analysis_result_if_available(
            'area_bins', self, np.arange(1, 35))
        distanceBins = self.dataSet.load_numpy_analysis_result_if_available(
            'distance_bins', self,
            np.arange(
                0, decodeTask.parameters['distance_threshold']+0.02, 0.01))
        intensityBins = self.dataSet.load_numpy_analysis_result_if_available(
            'intensity_bins', self, None)

        blankCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'blank_counts', self, None)
        codingCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'coding_counts', self, None)

        self.dataSet.save_numpy_analysis_result(
            areaBins, 'area_bins', self)
        self.dataSet.save_numpy_analysis_result(
            distanceBins, 'distance_bins', self)

        updated = False
        while not all(completeFragments):
            if (intensityBins is None or
                    blankCounts is None or codingCounts is None):
                for i in range(self.fragment_count()):
                    if not pendingFragments[i] and decodeTask.is_complete(i):
                        pendingFragments[i] = decodeTask.is_complete(i)

                if np.sum(pendingFragments) >= min(20, self.fragment_count()):
                    def extreme_values(inputData: pandas.Series):
                        return inputData.min(), inputData.max()
                    sampledFragments = np.random.choice(
                            [i for i, p in enumerate(pendingFragments) if p],
                            size=20)
                    intensityExtremes = [
                        extreme_values(barcodeDB.get_barcodes(
                            i, columnList=['mean_intensity'])['mean_intensity'])
                        for i in sampledFragments]
                    maxIntensity = np.log10(
                            np.max([x[1] for x in intensityExtremes]))
                    intensityBins = np.arange(0, 2 * maxIntensity,
                                              maxIntensity / 100)
                    self.dataSet.save_numpy_analysis_result(
                        intensityBins, 'intensity_bins', self)

                    blankCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))
                    codingCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))

            else:
                for i in range(self.fragment_count()):
                    if not completeFragments[i] and decodeTask.is_complete(i):
                        barcodes = barcodeDB.get_barcodes(
                            i, columnList=['barcode_id', 'mean_intensity',
                                           'min_distance', 'area'])
                        blankCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_blank_indexes())],
                            intensityBins, distanceBins, areaBins)
                        codingCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_coding_indexes())],
                            intensityBins, distanceBins, areaBins)
                        updated = True
                        completeFragments[i] = True

                if updated:
                    self.dataSet.save_numpy_analysis_result(
                        completeFragments, 'complete_fragments', self)
                    self.dataSet.save_numpy_analysis_result(
                        blankCounts, 'blank_counts', self)
                    self.dataSet.save_numpy_analysis_result(
                        codingCounts, 'coding_counts', self)


class AdaptiveFilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)

        bcDatabase.write_barcodes(adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes), fov=fragmentIndex)


class RemoveOverlapBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters potential ovrlapping barcodes
    due to dense imaging along the z axel.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'distance_cutoff' not in self.parameters:
            self.parameters['distance_cutoff'] = 1.1

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task'],
                self.parameters['filter_task']]

    def _run_analysis(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        
        distance_cutoff = self.parameters['distance_cutoff']

        bcDB = filterTask.get_barcode_database()
        barcodes = bcDB.get_barcodes(fragmentIndex)
        
        barcodeFiltered = []
        for barcode_id in barcodes.barcode_id.value_counts().index:
            y_sub = barcodes[barcodes.barcode_id == barcode_id]
            centroids = np.array([y_sub.global_x, y_sub.global_y, y_sub.global_z]).T
            if centroids.shape[0] < 2:
                barcodeFiltered.append(y_sub)
                continue

            n_neighbors = 2
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, 
                                algorithm='ball_tree').fit(centroids)
            distances, indices = nbrs.kneighbors(centroids)
    
            # create knn graph
            graph = np.zeros((centroids.shape[0], centroids.shape[0]), dtype=bool)
            np.fill_diagonal(graph, 1)
            for i in range(len(indices)):
                idx = indices[i][1]
                dst = distances[i][1]
                if dst < distance_cutoff:
                    graph[i,idx] = 1
    
            # identify connectied components
            n_components, newLabelList = connected_components(
                csgraph=csr_matrix(graph), directed=False, 
                return_labels=True)
    
            # randomly pick up one molecules=
            idxSel = np.array([ np.random.choice(np.where(newLabelList == j)[0]) \
                for j in range(n_components) ])
    
            # append barcodes
            barcodeFiltered.append(y_sub.iloc[idxSel])

        barcodeFiltered = pandas.concat(barcodeFiltered, axis=0)
        
        bcDatabase = self.get_barcode_database()
        bcDatabase.write_barcodes(barcodeFiltered, fov=fragmentIndex)
        

