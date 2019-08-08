import pandas
import numpy as np

from merlin.core import analysistask


class PartitionBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
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
        return [self.parameters['filter_task'],
                self.parameters['assignment_task']]

    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()]
            ).reset_index(drop=True)

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov)

    def _run_analysis(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_barcodes_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])

        codebook = self.dataSet.get_codebook()

        bcDB = filterTask.get_barcode_database()
        currentFOVBarcodes = bcDB.get_barcodes(fragmentIndex)
        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells),
                           self.dataSet.get_codebook().get_barcode_count())),
            columns=range(self.dataSet.get_codebook().get_barcode_count()),
            index=[x.get_feature_id() for x in currentCells])

        for cell in currentCells:
            contained = cell.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'global_z']].values)
            count = currentFOVBarcodes[contained].groupby('barcode_id').size()
            count = count.reindex(
                range(self.dataSet.get_codebook().get_barcode_count()),
                fill_value=0)
            countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['partition_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['partition_task'])
        parsedBarcodes = pTask.get_partitioned_barcodes()

        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'partition_barcodes_combined',
                    self.get_analysis_name())
