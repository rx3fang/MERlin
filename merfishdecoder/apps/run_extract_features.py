import os
import tifffile
import numpy as np
import geopandas as geo
import pandas as pd

from merfishdecoder.core import dataset
from merfishdecoder.util import utilities
from merfishdecoder.util import segmentation

def run_job(
    dataSetName: str = None,
    fov: int = None,
    zpos: float = None,
    segmentedImagesName: str = None,
    outputName: str = None):
    
    """
    Extract features from decoded images for a zplane fov.

    Args
    ----
    dataSetName: input dataset name.

    fov: field of view.
    
    zpos: z plane positions.

    segmentedImagesName: directory contains segmented images.
    
    outputName: output file name for segmented images.
    
    """
    
    # dataSetName = "MERFISH_test/data"
    # fov = 0
    # zpos = 3.0
    # segmentedImagesName = "segmentedImages/fov_1_zpos_3.0_DAPI.npz"
    # outputName = "segmentedFeatures/fov_1_zpos_3.0_DAPI"

    # check points
    utilities.print_checkpoint("Extract Features")
    utilities.print_checkpoint("Start")
    
    # create merfish dataset
    dataSet = dataset.MERFISHDataSet(
        dataSetName)

    # change to working directory
    os.chdir(dataSet.analysisPath)

    # create the folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)

    # extract the feature
    segmentedImage = np.load(segmentedImagesName)["mask"]
    ft = [ (idx, segmentation.extract_polygon_per_index(segmentedImage, idx)) \
        for idx in np.unique(segmentedImage[segmentedImage > 0]) ]
    ft = [ (i, x) for (i, x) in ft if x != None ]

    # convert to a data frame
    if len(ft) > 0:
        features = geo.GeoDataFrame(
            pd.DataFrame({
                "fov": [fov] * len(ft),
                "global_z": [zpos] * len(ft),
                "z": dataSet.get_z_positions().index(zpos)}),
            geometry=[x[1] for x in ft])
    else:
        features = geo.GeoDataFrame(
            pd.DataFrame(columns = ["fov", "global_z", "z", "x", "y"]), 
            geometry=None)

    if not features.empty:
        features = features.assign(x = features.centroid.x)
        features = features.assign(y = features.centroid.y)
        features.to_file(outputName)
    else:
        with open(outputName, 'w') as fp: 
            pass

    utilities.print_checkpoint("Done")


