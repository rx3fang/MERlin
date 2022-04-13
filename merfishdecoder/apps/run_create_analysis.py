import merfishdecoder
from merfishdecoder.core import dataset
from merfishdecoder.util import utilities

def run_job(dataSetName: str = None,
            codebookName: str = None,
            dataOrganizationName: str = None,
            microscopeParameterName: str = None,
            positionName: str = None,
            microscopeChromaticAberrationName: str = None,
            dataHome: str = None,
            analysisHome: str = None):
    
    utilities.print_checkpoint("Create MERFISH analysis")
    utilities.print_checkpoint("Start")
    dataHome = merfishdecoder.DATA_HOME \
        if dataHome is None else dataHome 

    analysisHome = merfishdecoder.ANALYSIS_HOME \
        if analysisHome is None else analysisHome 

    dataSet = dataset.MERFISHDataSet(
        dataDirectoryName = dataSetName,
        codebookNames = [codebookName],
        dataOrganizationName = dataOrganizationName,
        positionFileName = positionName,
        dataHome = dataHome,
        analysisHome = analysisHome,
        microscopeParametersName = microscopeParameterName,
        microscopeChromaticAberrationName = microscopeChromaticAberrationName);

    utilities.print_checkpoint("Done")

