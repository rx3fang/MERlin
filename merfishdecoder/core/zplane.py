import os
import tifffile
import numpy as np
from copy import copy 
from merfishdecoder.util  import imagereader
from merfishdecoder.core  import dataset

class Frame(object):
    def __init__(self, 
        dataSet, 
        fov: int = None, 
        zpos: float = None,
        readoutName: str = None):

        """
        
        Create a new Frame object for a specific MERFISH dataset.
        
        Args:
            dataSetName: MERFISH dataset name.
            
            fov: An integer indicates the field of view.
            
            zpos: A float number indicates the zpos of the frame.
            
            readoutName: A string indicates the specific the frame name.
                Also known as frame name.
        
        """
        
        self._dataSet = dataSet;
        self._fiducial = None;
        self._img = None;
        self._zpos = zpos;
        self._fov = fov;
        self._readoutName = readoutName;
        
        bit_i = self._dataSet.dataOrganization.data["channelName"][
            int(np.where(self._dataSet.dataOrganization.data["readoutName"] == readoutName)[0])];

        self._fiducial_file_name = \
            self._dataSet.dataOrganization.get_fiducial_filename(
            self._dataSet.dataOrganization.get_data_channel_index(bit_i),
            fov);

        self._fiducial_frame_index = self._dataSet.dataOrganization.get_fiducial_frame_index(
            self._dataSet.dataOrganization.get_data_channel_index(bit_i));
            
        self._image_file_name = self._dataSet.dataOrganization.get_image_filename(
            self._dataSet.dataOrganization.get_data_channel_index(bit_i),
            fov);

        self._image_frame_index = self._dataSet.dataOrganization.get_image_frame_index(
            self._dataSet.dataOrganization.get_data_channel_index(bit_i),
            zpos);

        self._image_color = self._dataSet.dataOrganization.get_data_channel_color(
            self._dataSet.dataOrganization.get_data_channel_index(bit_i));
                
    def get_image_color(self):
        """
    
        Get frame color
    
        """
        return self._image_color

    def load_readout_raw_image(self):
        """
    
        Load readout raw image
    
        """
        movie = imagereader.infer_reader(
                self._dataSet.rawDataPortal.open_file(self._image_file_name));

        img = movie.load_frame(
            self._image_frame_index)

        if self._dataSet.transpose:
            img = np.transpose(img)
        if self._dataSet.flipHorizontal:
            img = np.flip(img, axis=1)
        if self._dataSet.flipVertical:
            img = np.flip(img, axis=0)
        self._img = img.copy()
        movie.close()

    def get_readout_image(self):
        """
    
        Get readout image
    
        """
        return self._img

    def load_fiducial_raw_image(self):
        """
    
        Load fiducial raw image
    
        """
        movie = imagereader.infer_reader(
                self._dataSet.rawDataPortal.open_file(
                self._fiducial_file_name));
        img = movie.load_frame(
            self._fiducial_frame_index)
        if self._dataSet.transpose:
            img = np.transpose(img)
        if self._dataSet.flipHorizontal:
            img = np.flip(img, axis=1)
        if self._dataSet.flipVertical:
            img = np.flip(img, axis=0)
        self._fiducial = img.copy()
        movie.close()
    
    def get_fiducial_image(self):
        """
    
        Get fiducial image
    
        """
        return self._fiducial
    
    def del_readout_raw_image(self):
        """
    
        Remove readout raw image
    
        """
        del self._img
        self._img = None

    def del_fiducial_raw_image(self):
        """
    
        Remove fiducial raw image
    
        """
        del self._fiducial
        self._img = None
        
class Zplane(object):
    
    """
    Purpuse: 
        A a plane object stores all the frames/images 
            from the same z position

    Args: 
        fov: an integer indicates the field of view that the 
            current zplane belongs to.
    
        zpos: z position

        readoutNames: A list of string indicate the frames
            to be included.        

    Returns: 
        Return a Zplane object
        
    """

    def __init__(self, 
                 dataSetName: str = None,
                 fov: int = None,
                 zpos: float = None):
        
        """
        
        Create a new Zplane object from a MERFISH dataset.
        
        Args:
            dataSetName: MERFISH dataset name.
            
            fov: An integer indicates the field of view.
            
            zpos: A float number indicates the zpos of the frame.
            
        """
                 
        self._dataSet = dataset.MERFISHDataSet(
            dataDirectoryName = dataSetName);
        readoutNames = \
            self._dataSet.dataOrganization.data["readoutName"]
        self._frames = dict(zip(readoutNames, 
            [ Frame(self._dataSet, fov = fov,
            zpos = zpos, readoutName = x) \
            for x in readoutNames ]))
        self._fov = fov
        self._zpos = zpos
        os.chdir(self._dataSet.analysisPath)
        
    def get_data_path(self) -> str:
    
        """
        
        Get data path
        
        """
    
        return self._dataSet.rawDataPath
    
    def get_analysis_path(self) -> str:
    
        """
        
        Get analysis path
        
        """
    
        return self._dataSet.analysisPath
    
    def get_stage_position(self) -> tuple:
    
        """
        
        Get stage position for current fov
        
        """
    
        return self._dataSet.get_stage_positions().iloc[self._fov]

    def get_image_size(self) -> tuple:
    
        """
        
        Get total number of frames in one zstack
        
        """

        return tuple(self._dataSet.get_image_dimensions())

    def get_film_size(self) -> tuple:
    
        """
        
        Get total number of frames in one zstack
        
        """

        (M, N) = self.get_image_size();
        return (len(self._frames), M, N);

    def get_microns_per_pixel(self) -> tuple:
    
        """
        
        Get micron per pixel
        
        """

        return self._dataSet.micronsPerPixel 
    
    def get_fov(self) -> int:
        
        """
        
        Get field of view
        
        """
        
        return self._fov

    def get_z_position(self) -> float:
        
        """
        
        Get z positions
        
        """
        
        return self._zpos
    
    def get_codebook(self):
        
        """
        
        Get readout name for each frame.
        
        """
        
        return self._dataSet.get_codebook()
    
    def get_readout_name(self) -> list:
        
        """
        
        Get readout name for each frame.
        
        """
        
        return list(self._frames.keys())
    
    def get_frame_by_readout_name(self, 
                                  readoutName: str = None
                                  ) -> Frame:

        """
        
        Get a frame object by readout name.
        
        """
        return self._frames[readoutName]                          
    
    def get_image_color(self,
                        readoutNames: list = None
                        ) -> list:
        
        """
        
        Get color for each readout
        
        """
        
        readoutNames = self.get_readout_name() \
             if readoutNames is None else readoutNames
        
        return [ self.get_frame_by_readout_name(rn).get_image_color() \
            for rn in readoutNames ]

    def get_bit_name(self) -> list:
        
        """
        
        Get readout names for encoding frames.
        
        """
        
        return self.get_codebook().get_bit_names()

    def get_feature_name(self) -> list:
        
        """
        
        Get readout names for feature frames.
        
        """
        
        return list(set(self.get_readout_name()) - set(self.get_bit_name()))

    def get_bit_count(self) -> int:
        
        """
        
        Get readout name for each frame.
        
        """
        
        return self.get_codebook().get_bit_count()
    
    def get_image_file_name(self) -> list:
        
        """
        
        Get image file name for each frame.
        
        """
        
        return [ frame._image_file_name for frame in self._frames ]

    
    def get_frames(self, 
                   readoutNames: list = None
                   ) -> Frame:

        """
        
        Get multiple frames by readout names
        
        """
        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames
        
        return [ self.get_frame_by_readout_name(rn) \
            for rn in readoutNames ]           
    
    
    def del_frames(self, readoutNames: list = None):

        """
        
        Remove a frame based on the readout name.
        If readoutNames is None, all frames will be deleted.
        
        """
        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames

        for rn in readoutNames:
            self._frames.pop(rn, None)
    
    def load_readout_image_from_readout_name(self, 
                                             readoutName: str = None):
    
        """
        
        Load readout image from a selected readout name.
        
        """
        self._frames[readoutName].load_readout_raw_image();
    
    def load_readout_images(self, 
                            readoutNames: list = None):
    
        """
        
        Load readout images from a list readout names.
        
        """
        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames 
        
        for rn in readoutNames:
            self.load_readout_image_from_readout_name(rn)
    
    def get_readout_image_from_readout_name(self, 
                                            readoutName: str
                                            ) -> np.ndarray:
        
        """
        
        Get a readout image from a readout name.
        
        """
        
        return self._frames[
            readoutName
            ].get_readout_image()
    
    def get_readout_images(self,
                           readoutNames: list=None):    

        """
        
        Get readout images from a list of readout names
        
        """

        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames 
        
        return np.array([ 
            self.get_readout_image_from_readout_name(rn) \
            for rn in readoutNames ])
        
    
    def load_fiducial_image_from_readout_name(self, 
                                              readoutName: str = None):
    
        """
        
        Load fiducial image from readout name.
        
        """

        self._frames[readoutName].load_fiducial_raw_image();
    
    def load_fiducial_images(self, 
                             readoutNames: list = None):
    
        """
        
        Load fiducial images from a list of readout names.
        
        """
        if readoutNames == None:
            readoutNames = self.get_readout_name()
        
        for rn in readoutNames:
            self.load_fiducial_image_from_readout_name(rn)
    
    def get_fiducial_image_from_readout_name(self, 
                                             readoutName: str = None
                                             ) -> np.ndarray:

        """
        
        Get a fiducial image from a given readout name.
        
        """

        return self._frames[
            readoutName].get_fiducial_image()
    
    def get_fiducial_images(self,
                            readoutNames: list= None
                            ) -> np.ndarray:    

        """
        
        Get fiducial images from a list of readout names.
        
        """
        
        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames 
        
        return np.array([ 
            self.get_fiducial_image_from_readout_name(rn) \
            for rn in readoutNames ])
    
    def del_readout_image_from_readout_name(self, 
                                            readoutName: str = None):
    
        """
        
        delete readout image from a readout name.
        
        """
        self._frames[readoutName].del_readout_raw_image();
    
    def del_readout_images(self,
                           readoutNames: list = None):
    
        """
        
        Delete readout images from a list readout names.
        
        """

        readoutNames = self.get_readout_name() \
            if readoutNames is None else readoutNames
    
        for rn in readoutNames:
            self.del_readout_image_from_readout_name(rn)
    
    
    def del_fiducial_image_from_readout_name(self, 
                                             readoutName: str = None):
    
        """
        
        Delete a fiducial raw image from a readout name.
        
        """
        self._frames[readoutName].del_fiducial_raw_image();
    
    def del_fiducial_images(self,
                            readoutNames: list = None):
    
        """
        
        delete fiducial raw image from readout name.
        
        """
        readoutNames =  self.get_readout_name() \
            if readoutNames is None else readoutNames
    
        for rn in readoutNames:
            self.del_fiducial_image_from_readout_name(rn);
    
    def save_readout_images(self, 
                            fileName: str = None, 
                            readoutNames: list = None):

        """
        
        Save readout images into a tif file
        
        """
        tifffile.imwrite(file = fileName, 
                data = self.get_readout_images(
                    self.get_readout_name()      \
                    if readoutNames is None else \
                    readoutNames).astype(np.uint16))

    def save_fiducial_images(self, 
                             fileName: str = None, 
                             readoutNames: list = None):

        """
        
        Save fiducial images into a tif file
        
        """

        tifffile.imwrite(file = fileName, 
                data = self.get_fiducial_images(
                    self.get_readout_name()      \
                    if readoutNames is None else \
                    readoutNames).astype(np.uint16))

    def get_chromatic_aberration_profile(self):

        """
        
        Get chromatic abberation correction profiles.
        
        """
        return \
            self._dataSet.get_chromatic_aberration_profile()

    def load_warped_images(self, filename):

        """
        
        Load warped images.
        
        """
        
        movie = tifffile.imread(filename)
        frameNames = self.get_readout_name()
        for fn in frameNames:
            self._frames[fn]._img = \
                movie[frameNames.index(fn)].copy()
        del movie

    def load_processed_images(self, filename):

        """
        
        Load processed images.
        
        """
        
        if filename.endswith("npz"):
            movie = np.load(filename)
            movie = movie["arr_0"]
        elif filename.endswith('.npy'):
            movie = np.load(filename)
            movie = movie["arr_0"]
        elif filename.endswith('.tif'):
            movie = tifffile.imread(filename)

        frameNames = self.get_bit_name()
        for fn in frameNames:
            self._frames[fn]._img = \
                movie[frameNames.index(fn)].copy()
        del movie

    
