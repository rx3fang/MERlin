#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to assign identified barcodes to features for each FOV seperately.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# latest update: 04/15/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------
# example
# python run_barcode_assignment_fov.py \
#        2021_04_07/data/ \
#        filteredBarcodes/fov_100.h5 \
#        exportedFeatures/DAPI/fov_100.shp 100 \
#        assignedBarcodes/barcodes_DAPI.h5 2

import sys
import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
import geopandas as geo

from shapely.geometry import Point, Polygon

from merfishdecoder.util import utilities
from merfishdecoder.core import dataset

def read_barcodes_per_fov(
    fname: str = None,
    fov: int = None):
    """
    Read barcode for a single FOV from a hdf5 file.
    -----------
    fname: a hdf file contains the barcodes (str)
    fov: fov number (int)
    """

    try:
        return pd.concat([
            pd.read_hdf(fname, key="fov_%d" % fov) ],
            axis=1)
    except KeyError:
        print("barcodes in fov_%d does not exist" % fov)
        return None

def assign_barcodes_per_fov(
    barcodeFileName: str = None,
    features: geo.geodataframe.GeoDataFrame = None,
    fov: int = None
    ) -> pd.DataFrame:

    """
    Assign barcodes to feature for each FOV.
    """

    # read barcodes from one FOV
    bd = read_barcodes_per_fov(
        fname = barcodeFileName,
        fov = fov)

    if bd is None:
        return None
    
    # extract features belong to one fov
    ft = features[features.fov == fov].reset_index()

    # assign barcodes to segments
    bd = bd.assign(feature_name = "NA")
    if ft.shape[0] > 0:
        for i, s in ft.iterrows():
            z = np.array(ft.global_z)[i]
            bdz = bd[bd.global_z == z]
            if bdz.shape[0] > 0:
                idxes = np.array([j for (x, y, j) in
                    zip(bdz["global_x"],
                        bdz["global_y"],
                        bdz.index) \
                        if Point(x,y).within(s.geometry)])
                if len(idxes) > 0:
                    bd.loc[idxes,"feature_name"] = s["name"]
    return bd


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

        """
        Assign barcode to features.

        Args
        ----
        dataSetName: input dataset name.
    
        exportedBarcodesName: exported barcode file name in .h5 format. 

        exportedFeaturesName: exported feature file name in .shp format.
        
        fov: an integer indicates the FOV to assign barcodes.
    
        outputName: output file name.
    
        maxCores: max number of processors for parallel computing. 
    
        """
        
        
        utilities.print_checkpoint(self.to_string() + "\n")
        utilities.print_checkpoint("Assign Barcode to Features")
        utilities.print_checkpoint("Start")
    
        # generate MERFISH dataset object
        dataSet = dataset.MERFISHDataSet(
                self.dataSetName)
    
        # change to work directory
        os.chdir(dataSet.analysisPath)
    
        # create output folder
        os.makedirs(os.path.dirname(self.outputName),
                exist_ok=True)

        # read features
        features = geo.read_file(
            self.exportedFeaturesName)
    
        # expand or shrink the feature
        features.geometry = \
            features.geometry.buffer(self.bufferSize)
    
        # assign barcode to feature for current FOV
        barcodes = assign_barcodes_per_fov(
            self.exportedBarcodesName, features, self.fov)
    
        # get rid of unassigned barcodes
        barcodes = barcodes[
            barcodes.feature_name != "NA"]
        
        # export assigned barcodes
        barcodes.to_hdf(
            self.outputName,
            key = "fov_%d" % self.fov)

        utilities.print_checkpoint("Done")

def main():
    dataSetName = sys.argv[1]
    exportedBarcodesName = sys.argv[2]
    exportedFeaturesName = sys.argv[3]
    fov = int(sys.argv[4])
    outputName = sys.argv[5]
    bufferSize = float(sys.argv[6])
    
    mt = merfishTask(
        dataSetName = dataSetName,
        exportedBarcodesName = exportedBarcodesName,
        exportedFeaturesName = exportedFeaturesName,
        fov = fov,
        outputName = outputName,
        bufferSize = bufferSize,
        maxCores = 1)

    mt.run_job()

if __name__ == "__main__":
    main()
