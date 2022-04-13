#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to combine all barcodes (hdf5) in a given folder.
# Barcodes will be indexed by the FOV.
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
import random 
import h5py

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import barcoder

class merfishTask:
    
    """
    A object for merfish task.

    Combine barcodes from different fov.

    """

    def __init__(self, **arguments):
        for (arg, val) in arguments.items():
            setattr(self, arg, val)

    def to_string(self):
        return ("\n".join(["%s = %s" % (str(key), str(val)) \
                for (key, val) in self.__dict__.items() ]))
    
    def run_job(self):
        dataSetName = self.dataSetName
        inputDir = self.inputDir
        outputFile = self.outputFile

        utilities.print_checkpoint(self.to_string() + "\n")
        utilities.print_checkpoint("Combine Barcodes")
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
        
        barcodeFileList = [ (fov, os.path.join(self.inputDir, "fov_%d.h5" % fov)) \
                for fov in dataSet.get_fovs() ]

        # check if expected file exists
        barcodeFileListExisted = [ (fov, fname) for (fov, fname) in barcodeFileList \
            if os.path.exists(fname) ]

        # read nad save barcodes to the output file
        for (fov, fname) in barcodeFileListExisted:
            bd = pd.read_hdf(fname)
            bd.to_hdf(self.outputFile, key="fov_%d" % fov)

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



