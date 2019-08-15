import os
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas
import merlin
plt.style.use(
        os.sep.join([os.path.dirname(merlin.__file__),
                     'ext', 'default.mplstyle']))
import seaborn
import numpy as np

from merlin.core import analysistask
from merlin.analysis import filterbarcodes
from merlin.util import binary

class PlotPerformance(analysistask.AnalysisTask):

    """
    An analysis task that generates plots depicting metrics of the MERFISH
    decoding.
    """

    # TODO all the plotting should be refactored. I do not like the way
    # this class is structured as a long list of plotting functions. It would
    # be more convenient if each plot could track it's dependent tasks and
    # be executed once those tasks are complete.

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        #TODO - move this definition to run_analysis()
        self.optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        self.decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        self.filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])
        if 'segment_task' in self.parameters:
            self.segmentTask = self.dataSet.load_analysis_task(
                    self.parameters['segment_task'])
        else:
            self.segmentTask = None

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 180

    def get_dependencies(self):
        return [self.parameters['decode_task'], self.parameters['filter_task']]

    def _plot_fpkm_correlation(self):
        fpkmPath = os.sep.join([merlin.FPKM_HOME, self.parameters['fpkm_file']])
        fpkm = pandas.read_csv(fpkmPath, index_col='name')
        barcodes = self.filterTask.get_barcode_database().get_barcodes()
        codebook = self.filterTask.get_codebook()

        barcodeIndexes = codebook.get_coding_indexes()
        barcodeCounts = np.array(
            [np.sum(barcodes['barcode_id'] == i) for i in barcodeIndexes])
        fpkmCounts = np.array(
            [fpkm.loc[codebook.get_name_for_barcode_index(i)]['FPKM'] for
             i in barcodeIndexes])

        fig = plt.figure(figsize=(4, 4))
        plt.loglog(fpkmCounts, barcodeCounts, '.', alpha=0.5)
        plt.ylabel('Detected counts')
        plt.xlabel('FPKM')
        correlation = np.corrcoef(
            np.log(fpkmCounts + 1), np.log(barcodeCounts + 1))
        plt.title('%s (r=%0.2f)' % (self.parameters['fpkm_file'],
                                    correlation[0, 1]))
        self.dataSet.save_figure(self, fig, 'fpkm_correlation')

    def _plot_radial_density(self):
        bitColors = self.dataSet.get_data_organization().data['color']
        bcSet = self.filterTask.get_codebook().get_barcodes()
        singleColorBarcodes = [i for i, b in enumerate(bcSet) if
                               bitColors[np.where(b)[0]].nunique() == 1]
        multiColorBarcodes = [i for i, b in enumerate(bcSet) if
                              bitColors[np.where(b)[0]].nunique() > 1]

        barcodes = self.filterTask.get_barcode_database().get_barcodes()
        sBC = barcodes[barcodes['barcode_id'].isin(singleColorBarcodes)]
        mBC = barcodes[barcodes['barcode_id'].isin(multiColorBarcodes)]

        imageSize = self.dataSet.get_image_dimensions()
        width = imageSize[0]
        height = imageSize[1]

        def radial_distance(x, y):
            return np.sqrt((x - 0.5 * width) ** 2 + (y - 0.5 * height) ** 2)

        sRD = [radial_distance(x, y) for x, y in zip(sBC['x'], sBC['y'])]
        mRD = [radial_distance(x, y) for x, y in zip(mBC['x'], mBC['y'])]

        fig = plt.figure(figsize=(7, 7))

        countsS, binsS = np.histogram(sRD, bins=np.arange(0, 1000, 5))
        radialCountsS = countsS[1:]
        normS = radialCountsS / np.mean(radialCountsS[:20])
        plt.plot(binsS[1:-1], normS)

        countsM, binsM = np.histogram(mRD, bins=np.arange(0, 1000, 5))
        radialCountsM = countsM[1:]
        normM = radialCountsM / np.mean(radialCountsM[:20])
        plt.plot(binsM[1:-1], normM)
        plt.legend(['Single color barcodes', 'Multi color barcodes'])
        plt.xlabel('Radius')
        plt.ylabel('Normalized radial barcode density')
        self.dataSet.save_figure(self, fig, 'barcode_radial_density')

    # TODO - the functions in this class have too much repeated code
    # TODO - for the following 4 plots, I can add a line indicating the
    # barcode selection thresholds.
    def _plot_barcode_intensity_distribution(self):
        bcIntensities = self.decodeTask.get_barcode_database() \
                .get_barcode_intensities()
        fig = plt.figure(figsize=(4, 4))
        plt.hist(np.log10(bcIntensities), bins=500)
        plt.xlabel('Mean intensity ($log_{10}$)')
        plt.ylabel('Count')
        plt.title('Intensity distribution for all barcodes')
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_intensity_distribution')

    def _plot_barcode_area_distribution(self):
        bcAreas = self.decodeTask.get_barcode_database() \
                .get_barcode_areas()
        fig = plt.figure(figsize=(4, 4))
        plt.hist(bcAreas, bins=np.arange(15))
        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Count')
        plt.title('Area distribution for all barcodes')
        plt.xticks(np.arange(15))
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_area_distribution')

    def _plot_barcode_distance_distribution(self):
        bcDistances = self.decodeTask.get_barcode_database() \
                .get_barcode_distances()
        fig = plt.figure(figsize=(4,4))
        plt.hist(bcDistances, bins=500)
        plt.xlabel('Barcode distance')
        plt.ylabel('Count')
        plt.title('Distance distribution for all barcodes')
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_distance_distribution')

    def _plot_barcode_intensity_area_violin(self):
        barcodeDB = self.decodeTask.get_barcode_database()
        intensityData = [np.log10(
            barcodeDB.get_intensities_for_barcodes_with_area(x).tolist())
                    for x in range(1, 15)]
        nonzeroIntensities = [x for x in intensityData if len(x) > 0]
        nonzeroIndexes = [i+1 for i, x in enumerate(intensityData)
                          if len(x) > 0]
        fig = plt.figure(figsize=(8, 4))
        plt.violinplot(nonzeroIntensities, nonzeroIndexes, showextrema=False,
                       showmedians=True)
        if not isinstance(
                self.filterTask, filterbarcodes.AdaptiveFilterBarcodes):
            plt.axvline(x=self.filterTask.parameters['area_threshold']-0.5,
                        color='green', linestyle=':')
            plt.axhline(y=np.log10(
                self.filterTask.parameters['intensity_threshold']),
                    color='green', linestyle=':')
        else:
            adaptiveThresholds = [a for a in
                                  self.filterTask.get_adaptive_thresholds()
                                  for _ in (0, 1)]
            adaptiveXCoords = [0.5] + [x for x in np.arange(
                1.5, len(adaptiveThresholds)/2) for _ in (0, 1)] \
                + [len(adaptiveThresholds)/2+0.5]
            plt.plot(adaptiveXCoords, adaptiveThresholds)

        plt.xlabel('Barcode area (pixels)')
        plt.ylabel('Mean intensity ($log_{10}$)')
        plt.title('Intensity distribution by barcode area')
        plt.xlim([0, 15])
        plt.tight_layout(pad=0.2)
        self.dataSet.save_figure(self, fig, 'barcode_intensity_area_violin')

    def _plot_bitwise_intensity_violin(self):
        bcDF = pandas.DataFrame(self.filterTask.get_codebook().get_barcodes())

        bc = self.filterTask.get_barcode_database().get_barcodes()
        bitCount = self.filterTask.get_codebook().get_bit_count()
        onIntensities = [bc[bc['barcode_id'].isin(bcDF[bcDF[i] == 1].index)]
                         ['intensity_%i' % i].tolist() for i in range(bitCount)]
        offIntensities = [bc[bc['barcode_id'].isin(bcDF[bcDF[i] == 0].index)]
                          ['intensity_%i' % i].tolist() for i in
                          range(bitCount)]
        fig = plt.figure(figsize=(bitCount / 2, 5))
        onViolin = plt.violinplot(onIntensities,
                                  np.arange(1, bitCount + 1) - 0.25,
                                  showextrema=False, showmedians=True,
                                  widths=0.35)
        offViolin = plt.violinplot(offIntensities,
                                   np.arange(1, bitCount + 1) + 0.25,
                                   showextrema=False, showmedians=True,
                                   widths=0.35)
        plt.xticks(np.arange(1, bitCount + 1))
        plt.xlabel('Bit')
        plt.ylabel('Normalized intensity')
        plt.title('Bitwise intensity distributions')
        plt.legend([onViolin['bodies'][0], offViolin['bodies'][0]], ['1', '0'])

        self.dataSet.save_figure(self, fig, 'barcode_bitwise_intensity_violin')

    def _plot_blank_distribution(self):
        codebook = self.filterTask.get_codebook()
        bc = self.filterTask.get_barcode_database().get_barcodes()
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        blankIDs = codebook.get_blank_indexes()
        blankBC = bc[bc['barcode_id'].isin(blankIDs)]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(blankBC['global_x'], blankBC['global_y'],
            bins=(np.ceil(maxX-minX)/5, np.ceil(maxY-minY)/5),
            cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of blank barcodes')
        self.dataSet.save_figure(self, fig, 'blank_spatial_distribution')

    def _plot_matched_barcode_distribution(self):
        codebook = self.filterTask.get_codebook()
        bc = self.filterTask.get_barcode_database().get_barcodes()
        minX = np.min(bc['global_x'])
        minY = np.min(bc['global_y'])
        maxX = np.max(bc['global_x'])
        maxY = np.max(bc['global_y'])

        codingIDs = codebook.get_coding_indexes()
        codingBC = bc[bc['barcode_id'].isin(codingIDs)]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        h = ax.hist2d(codingBC['global_x'], codingBC['global_y'],
            bins=(np.ceil(maxX-minX)/5, np.ceil(maxY-minY)/5),
            cmap=plt.get_cmap('Greys'))
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label('Spot count', rotation=270)
        ax.set_aspect('equal', 'datalim')
        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Spatial distribution of identified barcodes')
        self.dataSet.save_figure(self, fig, 'barcode_spatial_distribution')

    def _plot_cell_segmentation(self):
        cellBoundaries = self.segmentTask.get_cell_boundaries()

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        def plot_cell_boundary(boundary):
            ax.plot([x[0] for x in boundary], [x[1] for x in boundary])
        cellPlots = [plot_cell_boundary(b) for b in cellBoundaries]

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Cell boundaries')
        self.dataSet.save_figure(self, fig, 'cell_boundaries')

    def _plot_optimization_scale_factors(self):
        fig = plt.figure(figsize=(5,5))
        seaborn.heatmap(self.optimizeTask.get_scale_factor_history())
        plt.xlabel('Bit index')
        plt.ylabel('Iteration number')
        plt.title('Scale factor optimization history')
        self.dataSet.save_figure(self, fig, 'optimization_scale_factors')

    def _plot_optimization_barcode_counts(self):
        fig = plt.figure(figsize=(5,5))
        seaborn.heatmap(self.optimizeTask.get_barcode_count_history())
        plt.xlabel('Barcode index')
        plt.ylabel('Iteration number')
        plt.title('Barcode counts optimization history')
        self.dataSet.save_figure(self, fig, 'optimization_barcode_counts')

    def _plot_barcode_abundances(self, barcodes, outputName):
        codebook = self.filterTask.get_codebook()
        blankIDs = codebook.get_blank_indexes()

        uniqueBarcodes, bcCounts = np.unique(barcodes['barcode_id'],
                                             return_counts=True)
        sortedIndexes = np.argsort(bcCounts)[::-1]

        fig = plt.figure(figsize=(12, 5))
        plt.bar(np.arange(len(bcCounts)),
                height=np.log10([bcCounts[x] for x in sortedIndexes]),
                width=1, color=(0.2, 0.2, 0.2))
        plt.bar([i for i, x in enumerate(sortedIndexes) if
                 uniqueBarcodes[x] in blankIDs],
                height=np.log10([bcCounts[x] for x in sortedIndexes if
                                 uniqueBarcodes[x] in blankIDs]),
                width=2, color='r')
        plt.xlabel('Sorted barcode index')
        plt.ylabel('Count (log10)')
        plt.title('Abundances for coding (gray) and blank (red) barcodes')

        self.dataSet.save_figure(self, fig, outputName)

    def _plot_all_barcode_abundances(self):
        bc = self.decodeTask.get_barcode_database().get_barcodes()
        self._plot_barcode_abundances(bc, 'all_barcode_abundances')

    def _plot_filtered_barcode_abundances(self):
        bc = self.filterTask.get_barcode_database().get_barcodes()
        self._plot_barcode_abundances(bc, 'flitered_barcode_abundances')

    def _run_analysis(self):
        if 'fpkm_file' in self.parameters:
            self._plot_fpkm_correlation()
        self._plot_bitwise_intensity_violin()
        self._plot_radial_density()
        self._plot_barcode_intensity_distribution()
        self._plot_barcode_area_distribution()
        self._plot_barcode_distance_distribution()
        self._plot_barcode_intensity_area_violin()
        self._plot_blank_distribution()
        self._plot_matched_barcode_distribution()
        self._plot_optimization_scale_factors()
        self._plot_optimization_barcode_counts()
        self._plot_all_barcode_abundances()
        self._plot_filtered_barcode_abundances()
        if self.segmentTask is not None:
            self._plot_cell_segmentation()
        # TODO _ analysis run times
        # TODO - barcode correlation plots
        # TODO - alignment error plots - need to save transformation information
        # first
        # TODO - barcode size spatial distribution global and FOV average
        # TODO - barcode distance spatial distribution global and FOV average
        # TODO - barcode intensity spatial distribution global and FOV average
        # TODO - good barcodes/blanks per cell
