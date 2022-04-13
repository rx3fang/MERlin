import os
import json
import shutil
import pandas
import numpy as np
import tifffile
import importlib
import time
import logging
import pickle
import datetime
from matplotlib import pyplot as plt
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Optional
import h5py
import tables
import xmltodict

import merfishdecoder
from merfishdecoder.util import imagereader
from merfishdecoder.util import dataportal
from merfishdecoder.data import codebook
from merfishdecoder.data import dataorganization

TaskOrName = str

class DataFormatException(Exception):
    pass

class DataSet(object):

    def __init__(self, 
                 dataDirectoryName: str,
                 dataHome: str = None, 
                 analysisHome: str = None,
                 microscopeParametersHome: str = None
                 ):
    
        """Create a dataset for the specified raw data.
        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersHome: the base path for storing microscope parameter files. 

        """
        if dataHome is None:
            dataHome = merfishdecoder.DATA_HOME
        else:
            merfishdecoder.DATA_HOME = dataHome
        
        if analysisHome is None:
            analysisHome = merfishdecoder.ANALYSIS_HOME
        else:
            merfishdecoder.ANALYSIS_HOME = analysisHome

        if microscopeParametersHome is None:
            microscopeParametersHome = merfishdecoder.MICROSCOPE_PARAMETERS_HOME
        else:
            merfishdecoder.MICROSCOPE_PARAMETERS_HOME = microscopeParametersHome

        self.dataSetName = dataDirectoryName
        self.dataHome = dataHome
        self.analysisHome = analysisHome

        self.rawDataPath = os.sep.join([dataHome, dataDirectoryName])
        self.rawDataPortal = dataportal.DataPortal.create_portal(
            self.rawDataPath)

        if not self.rawDataPortal.is_available():
            print('The raw data is not available at %s'.format(
                self.rawDataPath))

        self.analysisPath = os.sep.join([analysisHome, dataDirectoryName])
        os.makedirs(self.analysisPath, exist_ok=True)

        self.logPath = os.sep.join([self.analysisPath, 'logs'])
        os.makedirs(self.logPath, exist_ok=True)
        
        self._store_dataset_metadata()
        
    def _store_dataset_metadata(self) -> None:
        try:
            oldMetadata = self.load_json_analysis_result('dataset', None)
            if not merfishdecoder.is_compatible(oldMetadata['merfishdecoder_version']):
                raise merfishdecoder.IncompatibleVersionException(
                    ('Analysis was performed on dataset %s with MERlin '
                     + 'version %s, which is not compatible with the current '
                     + 'MERlin version %s')
                    % (self.dataSetName, oldMetadata['version'],
                       merfishdecoder.version()))
        except FileNotFoundError:
            newMetadata = {
                'merfishdecoder_version': merfishdecoder.version(),
                'module': type(self).__module__,
                'class': type(self).__name__,
                'dataset_name': self.dataSetName,
                'creation_date': str(datetime.datetime.now())
            }
            self.save_json_analysis_result(newMetadata, 'dataset', None)

    def load_json_analysis_result(
            self, resultName: str, analysisName: str, resultIndex: int = None,
            subdirectory: str = None) -> Dict:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, '.json')
        with open(savePath, 'r') as f:
            return json.load(f)
    
    def save_json_analysis_result(
            self, analysisResult: Dict, resultName: str,
            analysisName: str, resultIndex: int = None,
            subdirectory: str = None) -> None:
        savePath = self._analysis_result_save_path(
            resultName, analysisName, resultIndex, subdirectory, '.json')
        with open(savePath, 'w') as f:
            json.dump(analysisResult, f)
    
    def _analysis_result_save_path(
            self, resultName: str, analysisTask: TaskOrName,
            resultIndex: int=None, subdirectory: str=None,
            fileExtension: str=None) -> str:

        saveName = resultName
        if resultIndex is not None:
            saveName += '_' + str(resultIndex)
        if fileExtension is not None:
            saveName += fileExtension

        if analysisTask is None:
            return os.sep.join([self.analysisPath, saveName])
        else:
            return os.sep.join([self.get_analysis_subdirectory(
                analysisTask, subdirectory), saveName])
    
    
    @staticmethod
    def analysis_tiff_description(sliceCount: int, frameCount: int) -> Dict:
        imageDescription = {'ImageJ': '1.47a\n',
                            'images': sliceCount*frameCount,
                            'channels': 1,
                            'slices': sliceCount,
                            'frames': frameCount,
                            'hyperstack': True,
                            'loop': False}
        return imageDescription
    
    def _analysis_image_name(self, analysisTask: TaskOrName,
                             imageBaseName: str, imageIndex: int) -> str:
        destPath = self.get_analysis_subdirectory(
                analysisTask, subdirectory='images')
        if imageIndex is None:
            return os.sep.join([destPath, imageBaseName+'.tif'])
        else:
            return os.sep.join([destPath, imageBaseName+str(imageIndex)+'.tif'])
    
    def _analysis_result_save_path(
            self, resultName: str, analysisTask: TaskOrName,
            resultIndex: int=None, subdirectory: str=None,
            fileExtension: str=None) -> str:

        saveName = resultName
        if resultIndex is not None:
            saveName += '_' + str(resultIndex)
        if fileExtension is not None:
            saveName += fileExtension

        if analysisTask is None:
            return os.sep.join([self.analysisPath, saveName])
        else:
            return os.sep.join([self.get_analysis_subdirectory(
                analysisTask, subdirectory), saveName])
    
    def list_analysis_files(self, analysisTask: TaskOrName = None,
                            subdirectory: str = None, extension: str = None,
                            fullPath: bool = True) -> List[str]:
        basePath = self._analysis_result_save_path(
            '', analysisTask, subdirectory=subdirectory)
        fileList = os.listdir(basePath)
        if extension:
            fileList = [x for x in fileList if x.endswith(extension)]
        if fullPath:
            fileList = [os.path.join(basePath, x) for x in fileList]
        return fileList
    
    def save_dataframe_to_csv(
            self, dataframe: pandas.DataFrame, resultName: str,
            analysisTask: TaskOrName = None, resultIndex: int = None,
            subdirectory: str = None, **kwargs) -> None:
        """Save a pandas data frame to a csv file stored in this dataset.
        If a previous pandas data frame has been save with the same resultName,
        it will be overwritten
        Args:
            dataframe: the data frame to save
            resultName: the name of the output file
            analysisTask: the analysis task that the dataframe should be
                saved under. If None, the dataframe is saved to the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
            **kwargs: arguments to pass on to pandas.to_csv
        """
        savePath = self._analysis_result_save_path(
                resultName, analysisTask, resultIndex, subdirectory, '.csv')

        with open(savePath, 'w') as f:
            dataframe.to_csv(f, **kwargs)

    def load_dataframe_from_csv(
            self, resultName: str, analysisTask: TaskOrName = None,
            resultIndex: int = None, subdirectory: str = None,
            **kwargs) -> Union[pandas.DataFrame, None]:
        """Load a pandas data frame from a csv file stored in this data set.
        Args:
            resultName:
            analysisTask:
            resultIndex:
            subdirectory:
            **kwargs:
        Returns:
            the pandas data frame
        Raises:
              FileNotFoundError: if the file does not exist
        """
        savePath = self._analysis_result_save_path(
                resultName, analysisTask, resultIndex, subdirectory, '.csv') \

        with open(savePath, 'r') as f:
            return pandas.read_csv(f, **kwargs)

class ImageDataSet(DataSet):

    def __init__(self, dataDirectoryName: str, 
                 dataHome: str = None,
                 analysisHome: str = None,
                 microscopeParametersName: str = None,
                 microscopeChromaticAberrationName: str = None
                 ):
        """Create a dataset for the specified raw data.
        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome)

        if microscopeParametersName is not None:
            self._import_microscope_parameters(
                microscopeParametersName)
        
        if microscopeChromaticAberrationName is not None:
            self._import_chromatic_aberration_profile(
                microscopeChromaticAberrationName)
            
        self._load_microscope_parameters()
        self._load_chromatic_aberration_profile()

    def get_image_file_names(self):
        return sorted(self.rawDataPortal.list_files(
            extensionList=['.dax', '.tif', '.tiff']))

    def load_image(self, imagePath, frameIndex):
        with imagereader.infer_reader(
                self.rawDataPortal.open_file(imagePath)) as reader:
            imageIn = reader.load_frame(int(frameIndex))
            if self.transpose:
                imageIn = np.transpose(imageIn)
            if self.flipHorizontal:
                imageIn = np.flip(imageIn, axis=1)
            if self.flipVertical:
                imageIn = np.flip(imageIn, axis=0)
            return imageIn 

    def image_stack_size(self, imagePath):
        """
        Get the size of the image stack stored in the specified image path.
        Returns:
            a three element list with [width, height, frameCount] or None
                    if the file does not exist
        """
        with imagereader.infer_reader(self.rawDataPortal.open_file(imagePath)
                                      ) as reader:
            return reader.film_size()

    def _import_microscope_parameters(self, microscopeParametersName):
        
        if microscopeParametersName is not None:
            if not os.path.exists(microscopeParametersName):
                sourcePath = os.sep.join(
                        [merfishdecoder.MICROSCOPE_PARAMETERS_HOME, 
                        microscopeParametersName])
            else:
                sourcePath = microscopeParametersName

        destPath = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])

        shutil.copyfile(sourcePath, destPath) 


    def _import_chromatic_aberration_profile(self, chromaticAberrationName):
        
        if chromaticAberrationName is not None:
            if not os.path.exists(chromaticAberrationName):
                sourcePath = os.sep.join(
                        [merfishdecoder.MICROSCOPE_PARAMETERS_HOME, 
                        chromaticAberrationName])
            else:
                sourcePath = chromaticAberrationName

        destPath = os.sep.join(
                [self.analysisPath, 'chromatic_aberration.pkl'])

        shutil.copyfile(sourcePath, destPath) 

    def _load_microscope_parameters(self): 
        path = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])
        
        if os.path.exists(path):
            with open(path) as inputFile:
                self.microscopeParameters = json.load(inputFile)
        else:
            self.microscopeParameters = {}

        self.flipHorizontal = self.microscopeParameters.get(
            'flip_horizontal', True)
        self.flipVertical = self.microscopeParameters.get(
            'flip_vertical', False)
        self.transpose = self.microscopeParameters.get('transpose', True)
        
        self.micronsPerPixel = self.microscopeParameters.get(
                'microns_per_pixel', 0.108)
        
        self.imageDimensions = self.microscopeParameters.get(
                'image_dimensions', [2048, 2048])

    def _load_chromatic_aberration_profile(self): 
        
        path = os.sep.join(
                [self.analysisPath, 'chromatic_aberration.pkl'])
        
        if os.path.exists(path):
            inputFile = open(path, "rb")
            self.chromaticAberrationProfile = pickle.load(inputFile)
            inputFile.close()
        else:
            self.chromaticAberrationProfile = {}

    def get_chromatic_aberration_profile(self):
        """Get the chromatic aberration profile."""

        return self.chromaticAberrationProfile

    def get_microns_per_pixel(self):
        """Get the conversion factor to convert pixels to microns."""

        return self.micronsPerPixel

    def get_image_dimensions(self):
        """Get the dimensions of the images in this data set.
        Returns:
            A tuple containing the width and height of each image in pixels.
        """
        return self.imageDimensions

    def get_image_xml_metadata(self, imagePath: str) -> Dict:
        """ Get the xml metadata stored for the specified image.
        Args:
            imagePath: the path to the image file (.dax or .tif)
        Returns: the metadata from the associated xml file
        """
        filePortal = self.rawDataPortal.open_file(
            imagePath).get_sibling_with_extension('.xml')
        return xmltodict.parse(filePortal.read_as_text())

class MERFISHDataSet(ImageDataSet):
    def __init__(self, dataDirectoryName: str, codebookNames: List[str] = None,
                 dataOrganizationName: str = None, positionFileName: str = None,
                 dataHome: str = None, analysisHome: str = None,
                 microscopeParametersName: str = None, 
                 microscopeChromaticAberrationName: str= None
                 ):
        """Create a MERFISH dataset for the specified raw data.
        Args:
            dataDirectoryName: the relative directory to the raw data
            codebookNames: A list of the names of codebooks to use. The codebook
                    should be present in the analysis parameters
                    directory. Full paths can be provided for codebooks
                    present other directories.
            dataOrganizationName: the name of the data organization to use.
                    The data organization should be present in the analysis
                    parameters directory. A full path can be provided for
                    a codebook present in another directory.
            positionFileName: the name of the position file to use.
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
            microscopeChromaticAberrationName: the name of the microscope
                    chromatic aberration profile.
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome,
                         microscopeParametersName, 
                         microscopeChromaticAberrationName)
        
        self.dataOrganization = dataorganization.DataOrganization(
                self, dataOrganizationName)
        if codebookNames:
            self.codebooks = [codebook.Codebook(self, name, i)
                              for i, name in enumerate(codebookNames)]
        else:
            self.codebooks = self.load_codebooks()

        if positionFileName is not None:
            self._import_positions(positionFileName)
        self._load_positions()

    def save_codebook(self, codebook: codebook.Codebook) -> None:
        """ Store the specified codebook in this dataset.
        If a codebook with the same codebook index and codebook name as the
        specified codebook already exists in this dataset, it is not
        overwritten.
        Args:
            codebook: the codebook to store
        Raises:
            FileExistsError: If a codebook with the same codebook index but
                a different codebook name is already save within this dataset.
        """
        existingCodebookName = self.get_stored_codebook_name(
            codebook.get_codebook_index())
        if existingCodebookName and existingCodebookName \
                != codebook.get_codebook_name():
            raise FileExistsError(('Unable to save codebook %s with index %i '
                                  + ' since codebook %s already exists with '
                                  + 'the same index')
                                  % (codebook.get_codebook_name(),
                                     codebook.get_codebook_index(),
                                     existingCodebookName))

        if not existingCodebookName:
            self.save_dataframe_to_csv(
                codebook.get_data(),
                '_'.join(['codebook', str(codebook.get_codebook_index()),
                          codebook.get_codebook_name()]), index=False)
        
        if not existingCodebookName:
            self.save_dataframe_to_csv(
                codebook.get_data(),
                '_'.join(['codebook', str(codebook.get_codebook_index()),
                          codebook.get_codebook_name()]), index=False)

    def load_codebooks(self) -> List[codebook.Codebook]:
        """ Get all the codebooks stored within this dataset.
        Returns:
            A list of all the stored codebooks.
        """
        codebookList = []

        currentIndex = 0
        currentCodebook = self.load_codebook(currentIndex)
        while currentCodebook is not None:
            codebookList.append(currentCodebook)
            currentIndex += 1
            currentCodebook = self.load_codebook(currentIndex)

        return codebookList

    def load_codebook(self, codebookIndex: int = 0
                      ) -> Optional[codebook.Codebook]:
        """ Load the codebook stored within this dataset with the specified
        index.
        Args:
            codebookIndex: the index of the codebook to load.
        Returns:
            The codebook stored with the specified codebook index. If no
            codebook exists with the specified index then None is returned.
        """
        codebookFile = [x for x in self.list_analysis_files(extension='.csv')
                        if ('codebook_%i_' % codebookIndex) in x]
        if len(codebookFile) < 1:
            return None
        codebookName = '_'.join(os.path.splitext(os.path.basename(
            codebookFile[0]))[0].split('_')[2:])
        return codebook.Codebook(
            self, codebookFile[0], codebookIndex, codebookName)

    def get_stored_codebook_name(self, codebookIndex: int = 0) -> Optional[str]:
        """ Get the name of the codebook stored within this dataset with the
        specified index.
        Args:
            codebookIndex: the index of the codebook to load to find the name
                of.
        Returns:
            The name of the codebook stored with the specified codebook index.
            If no codebook exists with the specified index then None is
            returned.
        """
        codebookFile = [x for x in self.list_analysis_files(extension='.csv')
                        if ('codebook_%i_' % codebookIndex) in x]
        if len(codebookFile) < 1:
            return None
        return '_'.join(os.path.splitext(
            os.path.basename(codebookFile[0]))[0].split('_')[2:])

    def get_codebooks(self) -> List[codebook.Codebook]:
        """ Get the codebooks associated with this dataset.
        Returns:
            A list containing the codebooks for this dataset.
        """
        return self.codebooks

    def get_codebook(self, codebookIndex: int = 0) -> codebook.Codebook:
        return self.codebooks[codebookIndex]

    def get_data_organization(self):
        return self.dataOrganization

    def _import_positions_from_metadata(self):
        positionData = []
        for f in self.get_fovs():
            metadata = self.get_image_xml_metadata(
                self.dataOrganization.get_image_filename(0, f))
            currentPositions = \
                metadata['settings']['acquisition']['stage_position']['#text'] \
                .split(',')
            positionData.append([float(x) for x in currentPositions])
        positionPath = os.sep.join([self.analysisPath, 'positions.csv'])
        np.savetxt(positionPath, np.array(positionData), delimiter=',')

    def _load_positions(self):
        positionPath = os.sep.join([self.analysisPath, 'positions.csv'])
        if not os.path.exists(positionPath):
            self._import_positions_from_metadata()
        self.positions = pandas.read_csv(
            positionPath, header=None, names=['X', 'Y'])

    def _read_positions(self, positionFileName):
        self.positions = pandas.read_csv(
            positionFileName, header=None, names=['X', 'Y'])

    def _import_positions(self, positionFileName):
        sourcePath = os.sep.join([merfishdecoder.POSITION_HOME, positionFileName])
        destPath = os.sep.join([self.analysisPath, 'positions.csv'])
        
        shutil.copyfile(sourcePath, destPath)    

    def _convert_parameter_list(self, listIn, castFunction, delimiter=';'):
        return [castFunction(x) for x in listIn.split(delimiter) if len(x)>0]
    
    def get_stage_positions(self) -> List[List[float]]:
        return self.positions

    def get_fov_offset(self, fov: int) -> Tuple[float, float]:
        """Get the offset of the specified fov in the global coordinate system.
        This offset is based on the anticipated stage position.
        Args:
            fov: index of the field of view
        Returns:
            A tuple specifying the x and y offset of the top right corner
            of the specified fov in pixels.
        """
        # TODO - this should be implemented using the position of the fov.
        return self.positions.loc[fov]['X'], self.positions.loc[fov]['Y']

    def z_index_to_position(self, zIndex: int) -> float:
        """Get the z position associated with the provided z index."""

        return self.get_z_positions()[zIndex]

    def position_to_z_index(self, zPosition: float) -> int:
        """Get the z index associated with the specified z position
        
        Raises:
             Exception: If the provided z position is not specified in this
                dataset
        """

        zIndex = np.where(self.get_z_positions() == zPosition)[0]
        if len(zIndex) == 0:
            raise Exception('Requested z=%0.2f position not found.' % zPosition)
        return zIndex[0]

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this dataset.
        Returns:
            A sorted list of all unique z positions
        """
        return self.dataOrganization.get_z_positions()
    
    def get_fovs(self) -> List[int]:
        return self.dataOrganization.get_fovs()

    def get_imaging_rounds(self) -> List[int]:
        # TODO - check this function
        return np.unique(self.dataOrganization.fileMap['imagingRound'])
    
    def get_raw_image(self, dataChannel, fov, zPosition):
        return self.load_image(
                self.dataOrganization.get_image_filename(dataChannel, fov),
                self.dataOrganization.get_image_frame_index(
                    dataChannel, zPosition))
    
    def get_fiducial_image(self, dataChannel, fov):
        return self.load_image(
                self.dataOrganization.get_fiducial_filename(dataChannel, fov),
                self.dataOrganization.get_fiducial_frame_index(dataChannel))
