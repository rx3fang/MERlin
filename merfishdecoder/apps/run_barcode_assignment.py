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

def run_job(
    dataSetName: str = None,
    exportedBarcodesName: str = None,
    exportedFeaturesName: str = None,
    outputName: str = None,
    bufferSize: float = 0,
    maxCores: int = 1):

    """
    Assign barcode to features.

    Args
    ----
    dataSetName: input dataset name.
    
    exportedBarcodesName: exported barcode file name in .h5 format. 

    exportedFeaturesName: exported feature file name in .shp format.
    
    outputName: output file name.
    
    maxCores: max number of processors for parallel computing. 
    
    """
    
    # dataSetName = "20200303_hMTG_V11_4000gene_best_sample/data"
    # exportedBarcodesName = "exportedBarcodes/barcodes.h5"
    # exportedFeaturesName = "exportedFeatures/polyT"
    # outputName = "assignedBarcodes/barcodes_polyT.h5"
    # maxCores = 10
    
    utilities.print_checkpoint("Assign Barcode to Features")
    utilities.print_checkpoint("Start")
    
    # generate MERFISH dataset object
    dataSet = dataset.MERFISHDataSet(
            dataSetName)
    
    # change to work directory
    os.chdir(dataSet.analysisPath)
    
    # create output folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)

    # read features
    features = geo.read_file(
        exportedFeaturesName)
    
    # expand or shrink the feature
    features.geometry = \
        features.geometry.buffer(bufferSize)

    # assign barcodes to cell
    with mp.Pool(processes=maxCores) as pool:
        barcodes = pd.concat(pool.starmap(
            assign_barcodes_per_fov, 
            (( exportedBarcodesName, features, fov) \
            for fov in dataSet.get_fovs())))
    
    # write it down
    barcodes = barcodes[
        barcodes.feature_name != "NA"]

    for fov in np.unique(barcodes.fov):
        barcodes.loc[
            barcodes.fov == fov].to_hdf(
            outputName,
            key = "fov_%d" % fov)
            
    utilities.print_checkpoint("Done")
