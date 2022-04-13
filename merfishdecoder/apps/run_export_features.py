#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to export identified features.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# lastest update: 04/14/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------
# example
# python run_export_features.py MERFISH_test/data exportedFeaturesFOV/DAPI/ exportedFeatures/DAPI.shp 
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


def FileCheck(fn):
    
    """
    Check whether a file is a valid shp file.
    """
    
    try:
        geo.read_file(fn);
        return True
    except:
        return False

def run_job(
    dataSetName: str = None,
    outputName: str = None,
    segmentedFeaturesDir: str = None):
    
    """
	Combine and export features for all FOVs.

    Args
    ----
    dataSetName: input dataset name.

    outputName: output file name for segmented images.

    """
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

    # load the segmentation files
    segmentedFeaturesName = \
        [ os.path.join(segmentedFeaturesDir, x) \
        for x in os.listdir(segmentedFeaturesDir) if "shp" in x]

    # check file formats first and remove empty files
    segmentedFeaturesNameValid = \
        [ x for x in segmentedFeaturesName if FileCheck(x) ]

    # load all the segmented features
    features = pd.concat([
            geo.read_file(f) for f in segmentedFeaturesNameValid],
        ignore_index=True)

    # global alingment
    if not features.empty:
        features[['fov', 'x', 'y', 'z', 'global_x',
            'global_y', 'global_z', 'name',
            'geometry']].to_file(
            filename = outputName)

    utilities.print_checkpoint("Done")
    return(0)

def main():
    dataSetName = sys.argv[1]
    segmentedFeaturesDir = sys.argv[2]
    outputName = sys.argv[3]

    run_job(
        dataSetName = dataSetName, 
        outputName = outputName,
        segmentedFeaturesDir = segmentedFeaturesDir)
 
if __name__ == "__main__":
    main()


