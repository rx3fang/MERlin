#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to filter identified barcodes.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# latest update: 04/14/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------
# example:
# python run_filter_barcodes_fov.py \
#   2021_04_07/data/
#   exportedBarcodes/barcodes.h5
#   filteredBarcodes/
#   30 1.0 True \
#   5 0.55 10
 
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

    Export barcodes.

    Args
    ----
    dataSetName: input dataset name.

    decodedBarcodesName: a list of decoded barcode file names.

    outputName: output file that contains decoded barcodes.

    
    """


    def __init__(self, **arguments):
        for (arg, val) in arguments.items():
            setattr(self, arg, val)

    def to_string(self):
        return ("\n".join(["%s = %s" % (str(key), str(val)) \
                for (key, val) in self.__dict__.items() ]))
    
    def run_job(self):
        dataSetName = self.dataSetName
        exportedBarcodesName = self.exportedBarcodesName
        outputName = self.outputName
        fovNum = self.fovNum
        misIdentificationRate = self.misIdentificationRate
        keepBlankBarcodes = self.keepBlankBarcodes
        areaThreshold = self.areaThreshold
        distanceThreshold = self.distanceThreshold
        magnitudeThreshold = self.magnitudeThreshold

        utilities.print_checkpoint(self.to_string() + "\n")
        
        utilities.print_checkpoint("Filter Barcodes")
        
        utilities.print_checkpoint("Start")
    
        # generate MERFISH dataset object
        dataSet = dataset.MERFISHDataSet(
            dataDirectoryName = self.dataSetName);

        # change to work directory
        os.chdir(dataSet.analysisPath)
    
        # create output folder
        os.makedirs(os.path.dirname(self.outputName),
                exist_ok=True)
    
        # randomly sample fovNum barcodes
        f = h5py.File(self.exportedBarcodesName, 'r')
        fovs = list(f.keys())
        f.close()
    
        barcodes = pd.concat([ 
            pd.read_hdf(self.exportedBarcodesName, key = key)
            for key in random.sample(fovs, min(len(fovs), self.fovNum)) ],
            ignore_index = True)
    
        # estimate likelihood cutoff
        likelihoodThreshold = \
            barcoder.estimate_barcode_threshold(
            barcodes = barcodes,
            codebook = dataSet.get_codebook(),
            cutoff = self.misIdentificationRate,
            bins = 200);
    
        # filter barcodes
        for fov in fovs:
            barcodes = pd.read_hdf(
                    self.exportedBarcodesName, 
                    key=fov)
        
            barcodes = barcoder.filter_barcodes(
                barcodes,
                dataSet.get_codebook(),
                likelihoodThreshold=likelihoodThreshold,
                keepBlankBarcodes=self.keepBlankBarcodes,
                minAreaSize=self.areaThreshold)
        
            barcodes = barcodes[
                    (barcodes.area >= self.areaThreshold) & \
                    (barcodes.distance <= self.distanceThreshold) & \
                    (barcodes.magnitude >= self.magnitudeThreshold)]

            barcodes.to_hdf(
                "%s/%s.h5" % (self.outputName, fov),
                key = fov);

        utilities.print_checkpoint("Done")

def main():
    dataSetName = sys.argv[1]
    exportedBarcodesName = sys.argv[2]
    outputName = sys.argv[3]
    fovNum = int(sys.argv[4])
    misIdentificationRate = float(sys.argv[5])	
    keepBlankBarcodes = bool(sys.argv[6])
    areaThreshold = int(sys.argv[7])
    distanceThreshold = float(sys.argv[8])
    magnitudeThreshold = float(sys.argv[9])
    
    mt = merfishTask(
        dataSetName = dataSetName,
        exportedBarcodesName = exportedBarcodesName,
        outputName = outputName,
        fovNum=fovNum,
        misIdentificationRate = misIdentificationRate,
        keepBlankBarcodes = keepBlankBarcodes,
        areaThreshold = areaThreshold,
        distanceThreshold = distanceThreshold,
        magnitudeThreshold = magnitudeThreshold) 
    
    mt.run_job()

if __name__ == "__main__":
    main()



