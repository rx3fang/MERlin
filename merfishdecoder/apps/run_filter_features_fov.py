#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to filter barcodes for each field of view.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# lastest update: 04/14/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------
# example
# ----------------------------------------------------------------------------------------


import sys
import os
import geopandas as geo
import pandas as pd
import numpy as np
import multiprocessing as mp

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import segmentation
    
def run_job(
    dataSetName: str = None,
    exportedFeaturesName: str = None,
    outputName: str = None,
    minZplane: int = 3,
    borderSize: int = 50):
    
    """
    Extract features from decoded images for each fov.

    Args
    ----
    dataSetName: input dataset name.

    fov: the field of view to be processed.
    
    outputName: output file name for segmented images.
    
    """
    
    # dataSetName = "MERFISH_test/data"
    # exportedFeaturesName = "exportedFeatures/DAPI.shp"
    # outputName = "exportedFeatures/DAPI.shp"
    # borderSize = 50
    # minZplane = 3
    
    # check points
    utilities.print_checkpoint("Export Features")
    utilities.print_checkpoint("Start")
    
    # create merfish dataset
    dataSet = dataset.MERFISHDataSet(
        dataSetName)

    # change to working directory
    os.chdir(dataSet.analysisPath)

    # create output folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)
    
    # load all the segmented features
    features = geo.read_file(
        exportedFeaturesName)

    # filter feautres
    features = pd.concat([ 
        segmentation.filter_features_per_fov(
            dataSet = dataSet,
            features = features,
            fov = fov,
            minZplane = minZplane,
            borderSize = borderSize) \
            for fov in np.unique(features.fov) ],
        ignore_index = True)

    if not features.empty:
        features[['fov', 'x', 'y', 'z', 'global_x', 
            'global_y', 'global_z', 'name', 
            'geometry']].to_file(
            filename = outputName)
    else:
        with open(outputName, "w") as fout:
            pass

    utilities.print_checkpoint("Done")


