#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to export identified barcodes for all FOVs.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# latest update: 04/14/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------


import sys
import os
import pickle
import pandas as pd
import numpy as np
import random 

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import barcoder

def run_job(dataSetName: str = None,
            decodedBarcodesDir: str = None,
            outputName: str = None):
    
    """
    Export barcodes.

    Args
    ----
    dataSetName: input dataset name.
    
    decodedBarcodesDir: directory contains decoded barcodes.

    outputName: output file that contains decoded barcodes. 
    
    """
    
    utilities.print_checkpoint("Export Barcodes")
    utilities.print_checkpoint("Start")
    
    # generate zplane object
    dataSet = dataset.MERFISHDataSet(
        dataDirectoryName = dataSetName);

    # change to work directory
    os.chdir(dataSet.analysisPath)
    
    # create the folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)

    for fov in dataSet.get_fovs():

        fnames = [os.path.join(decodedBarcodesDir, \
            "fov_%d_zpos_%0.1f.h5" % (fov, zpos)) \
            for zpos in dataSet.get_z_positions()]

        # check if the file exists
        fnames = [fn for fn in fnames if os.path.exists(fn)]
        
        # continue if no barcode files are discovered
        if len(fnames) == 0:
            continue

        barcodes = barcoder.export_barcodes(
            obj = dataSet,
            fnames = fnames)

        # write it down
        barcodes.to_hdf(
            outputName,
            key = "fov_%d" % fov);

    utilities.print_checkpoint("Done")

def main():
	dataSetName = sys.argv[1]
	decodedBarcodesDir = sys.argv[2]
	outputName = sys.argv[3]
	
	run_job(
		dataSetName = dataSetName,
		decodedBarcodesDir = decodedBarcodesDir,
		outputName = outputName)	

if __name__ == "__main__":
	main()
