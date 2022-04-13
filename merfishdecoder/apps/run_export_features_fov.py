#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# An application to export identified features for each fild of view.
# ----------------------------------------------------------------------------------------
# Rongxin Fang
# lastest update: 04/14/21
# r4fang@gmail.com
# ----------------------------------------------------------------------------------------
# example
# python run_export_features_fov.py MERFISH_test/data 100 extractedFeaturesFOV/DAPI exportedFeatures/DAPI/feature_100.shp 20 
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
    segmentedFeaturesDir: str = None,
    bufferSize: int = 15):

    
    """
    Extract features from feature images.

    Args
    ----
    dataSetName: input dataset name.

    fov: the field of view to be processed.

    outputName: output file name for segmented images.

    """

    # dataSetName = "MERFISH_test/data"
    # segmentedFeaturesDir = "extractedFeatures/DAPI/"
    # outputName = "exportedFeatures/DAPI.shp"

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

    # connect features per fov
    features = pd.concat([
        segmentation.connect_features_per_fov(
            dataSet = dataSet,
            features = features,
            fov = fov,
            bufferSize = bufferSize) \
            for fov in np.unique(features.fov) ],
        ignore_index = True)

    # global alingment
    features = pd.concat([
        segmentation.global_align_features_per_fov(
            dataSet = dataSet,
            features = features,
            fov = fov) \
            for fov in np.unique(features.fov) ],
        ignore_index = True)

    if not features.empty:
        features[['fov', 'x', 'y', 'z', 'global_x',
            'global_y', 'global_z', 'name',
            'geometry']].to_file(
            filename = outputName)

    utilities.print_checkpoint("Done")

def run_job_per_fov(
    dataSetName: str = None,
    outputName: str = None,
    fov: str = None,
    segmentedFeaturesDir: str = None,
    bufferSize: int = 15):

    """
    Extract features from decoded images for each fov indepedently.

    Args
    ----
    dataSetName: input dataset name.

    outputName: output file name.

    fov: the field of view to be processed.
    
    segmentationFeatureDir: directory contains segmentation files.

    bufferSize: features in each zplane will be connected to 
        if their centroids overlap with each other.
        Before connecting, each centroid will be expanded by
        bufferSize to identify overlap.

    """


    # check points
    utilities.print_checkpoint("Export Features for fov %d\n" % fov)
    utilities.print_checkpoint("Start")

    # create merfish dataset
    dataSet = dataset.MERFISHDataSet(
        dataSetName)

    # change to working directory
    os.chdir(dataSet.analysisPath)

    # create output folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)
    
    # identify all segmentation files from the same fov
    segmentedFeaturesName = [ os.path.join(segmentedFeaturesDir, \
        "fov_%d_zpos_%0.1f_DAPI.shp" % (fov, zpos)) \
        for zpos in dataSet.get_z_positions() ]

    # check file formats first and remove empty files
    segmentedFeaturesNameValid = \
        [ x for x in segmentedFeaturesName if FileCheck(x) ]
    
    if len(segmentedFeaturesNameValid) == 0:
        return (0)
    
    # load all the segmented features
    features = pd.concat([
            geo.read_file(f) for f in segmentedFeaturesNameValid],
        ignore_index=True)

    # connect features per fov
    features = pd.concat([
        segmentation.connect_features_per_fov(
            dataSet = dataSet,
            features = features,
            fov = fov,
            bufferSize = bufferSize) \
            for fov in np.unique(features.fov) ],
        ignore_index = True)

    # global alingment
    features = pd.concat([
        segmentation.global_align_features_per_fov(
            dataSet = dataSet,
            features = features,
            fov = fov) \
            for fov in np.unique(features.fov) ],
        ignore_index = True)

    if not features.empty:
        features[['fov', 'x', 'y', 'z', 'global_x',
            'global_y', 'global_z', 'name',
            'geometry']].to_file(
            filename = outputName)

    utilities.print_checkpoint("Done")
    return(0)

def main():
    dataSetName = sys.argv[1]
    fov = int(sys.argv[2])
    segmentedFeaturesDir = sys.argv[3]
    outputName = sys.argv[4]
    bufferSize = int(sys.argv[5])

    run_job_per_fov(
        dataSetName = dataSetName, 
        outputName = outputName,
        segmentedFeaturesDir = segmentedFeaturesDir,
        fov = fov,
        bufferSize = bufferSize)
    
if __name__ == "__main__":
    main()


