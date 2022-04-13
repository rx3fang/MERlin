#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to combine all features (shp) in a given folder.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# latest update: 04/15/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------

import glob
import sys
import os
import pickle
import pandas as pd
import numpy as np
import geopandas as geo
import random 
import h5py

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import barcoder

class merfishTask:
    
    """
    A object for merfish task.

    """

    def __init__(self, **arguments):
        for (arg, val) in arguments.items():
            setattr(self, arg, val)

    def to_string(self):
        return ("\n".join(["%s = %s" % (str(key), str(val)) \
                for (key, val) in self.__dict__.items() ]))
    
    def run_job(self):

        utilities.print_checkpoint(self.to_string() + "\n")
        utilities.print_checkpoint("Combine Features")
        utilities.print_checkpoint("Start")
    
        # generate MERFISH dataset object
        dataSet = dataset.MERFISHDataSet(
            dataDirectoryName = self.dataSetName);

        # change to work directory
        os.chdir(dataSet.analysisPath)
    
        # create output folder
        os.makedirs(os.path.dirname(self.outputFile),
                exist_ok=True)
        
        # iterative every fov and report error if barcode
        # from the current fov does not exist in the 
        # given directory.
        barcodeFileList = [ (fov, os.path.join(self.inputDir, "fov_%d.shp" % fov)) \
                for fov in dataSet.get_fovs() ]

        # check if expected file exists
        barcodeFileListExisted = [ (fov, fname) for (fov, fname) in barcodeFileList \
            if os.path.exists(fname) ]

        # read nad save barcodes to the output file
        featureList = [ geo.read_file(fname) \
                for (fov, fname) in barcodeFileListExisted ]
        
        # combine all features
        features = pd.concat(featureList, ignore_index = True)
        features.to_file(self.outputFile)
        utilities.print_checkpoint("Done")

def main():
    dataSetName = sys.argv[1]
    inputDir = sys.argv[2]
    outputFile = sys.argv[3]
    
    mt = merfishTask(
        dataSetName = dataSetName,
        inputDir = inputDir,
        outputFile = outputFile)
    
    mt.run_job()
    
if __name__ == "__main__":
    main()



