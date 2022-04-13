import gc
import random
import cv2
import copy
import multiprocessing as mp
import SharedArray as sa
import tempfile
import os
import h5py

import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from skimage import measure

from merfishdecoder.data import codebook as cb

def extract_barcodes(
    decodedImage: np.ndarray = None,
    distanceImage: np.ndarray = None,
    probabilityImage: np.ndarray = None, 
    magnitudeImage: np.ndarray = None, 
    barcodesPerCore: int = 50,
    numCores: int = 1
    ) -> pd.core.frame.DataFrame:

    tmpPrefix = \
        tempfile.NamedTemporaryFile().name.split("/")[-1]

    decodedImageFile = \
        tmpPrefix + "_decodedImage"
    
    probabilityImageFile = \
        tmpPrefix + "_probabilityImage"

    distanceImageFile = \
        tmpPrefix + "_distanceImage"

    magnitudeImageFile = \
        tmpPrefix + "magnitudeImage"
    
    decodedImageShared = sa.create(
        "shm://" + decodedImageFile, 
        decodedImage.shape)
    
    probabilityImageShared = sa.create(
        "shm://" + probabilityImageFile, 
        probabilityImage.shape)

    magnitudeImageShared = sa.create(
        "shm://" + magnitudeImageFile, 
        magnitudeImage.shape)

    distanceImageShared = sa.create(
        "shm://" + distanceImageFile, 
        distanceImage.shape)
    
    decodedImageShared[:]     = decodedImage[:]
    probabilityImageShared[:] = probabilityImage[:]
    distanceImageShared[:]    = distanceImage[:]
    magnitudeImageShared[:]   = magnitudeImage[:]
    
    with mp.Pool(numCores) as process:
        barcodes = pd.concat(process.starmap(
            extract_barcodes_by_indexes, 
            [ ( decodedImageFile, probabilityImageFile, 
            distanceImageFile, magnitudeImageFile,  \
            np.arange(j, j + barcodesPerCore)) \
            for j in np.arange(
                0, decodedImage.max() + 1, 
                barcodesPerCore) ]), 
                ignore_index = True)
    
    sa.delete(decodedImageFile)
    sa.delete(probabilityImageFile)
    sa.delete(distanceImageFile)
    sa.delete(magnitudeImageFile)
    return barcodes

def extract_barcodes_by_indexes(
    decodedImageName: str = None, 
    probImageName: str = None,
    distImageName: str = None,
    magImageName: str = None,
    barcodeIndexes: np.ndarray = None
    ) -> pd.core.frame.DataFrame:
    
    o = sa.attach(decodedImageName)
    p = sa.attach(probImageName)
    m = sa.attach(magImageName)
    d = sa.attach(distImageName)

    propertiesO = measure.regionprops(
        measure.label(np.isin(o, barcodeIndexes)),
        intensity_image=o,
        cache=False)

    propertiesP = measure.regionprops(
        measure.label(np.isin(o, barcodeIndexes)),
        intensity_image=p,
        cache=False)

    propertiesD = measure.regionprops(
        measure.label(np.isin(o, barcodeIndexes)),
        intensity_image=d,
        cache=False)

    propertiesM = measure.regionprops(
        measure.label(np.isin(o, barcodeIndexes)),
        intensity_image=m,
        cache=False)
    
    if len(propertiesO) == 0:
        return pd.DataFrame(columns = 
            ["x", "y", 
            "x_max", "y_max",
            "x_start", "x_end", 
            "y_start", "y_end",
            "barcode_id", 
            "likelihood", 
            "magnitude", 
            "magnitude_min", 
            "magnitude_max", 
            "distance", 
            "distance_min", 
            "distance_max", 
            "area"])
    else:
        barcodeIDs = np.array(
            [prop.min_intensity for prop in propertiesO])

        xStart = np.array(
            [prop.bbox[0] for prop in propertiesO])

        xEnd = np.array(
            [prop.bbox[2] for prop in propertiesO])

        yStart = np.array(
            [prop.bbox[1] for prop in propertiesO])

        yEnd = np.array(
            [prop.bbox[3] for prop in propertiesO])
        
        centroidCoords = np.array(
            [prop.weighted_centroid for prop in propertiesP])
        centroids = centroidCoords[:, [1, 0]]              
        
        maxCoords = np.array(
            [ prop.coords[m[prop.coords[:,0], prop.coords[:,1]].argmax()] \
            for prop in propertiesP ]).astype(np.float64)
        maxCoords = maxCoords[:, [1, 0]]
        
        areas = np.array(
            [ x.area for x in propertiesP ]).astype(np.float)

        liks = np.array([ 
            -sum(np.log10(1 - x.intensity_image[x.image] + 1e-15)) \
            for x in propertiesP ]
            ).astype(np.float32)
        
        mags_mean = np.array([ 
            x.mean_intensity \
            for x in propertiesM ]
            ).astype(np.float32)

        mags_min = np.array([ 
            x.min_intensity \
            for x in propertiesM ]
            ).astype(np.float32)

        mags_max = np.array([ 
            x.max_intensity \
            for x in propertiesM ]
            ).astype(np.float32)

        dists_mean = np.array([ 
            x.mean_intensity \
            for x in propertiesD ]
            ).astype(np.float32)

        dists_min = np.array([ 
            x.min_intensity \
            for x in propertiesD ]
            ).astype(np.float32)
        
        dists_max = np.array([ 
            x.max_intensity \
            for x in propertiesD ]
            ).astype(np.float32)
        
        return pd.DataFrame({
            "x": centroids[:,0].astype(np.float),
            "y": centroids[:,1].astype(np.float),
            "x_max": maxCoords[:,0].astype(np.float),
            "y_max": maxCoords[:,1].astype(np.float),
            "x_start": xStart.astype(np.float),
            "x_end": xEnd.astype(np.float),
            "y_start": yStart.astype(np.float),
            "y_end": yEnd.astype(np.float),
            "barcode_id": barcodeIDs,
            "likelihood": liks,
            "magnitude": mags_mean,
            "magnitude_min": mags_min,
            "magnitude_max": mags_max,
            "distance": dists_mean,
            "distance_min": dists_min,
            "distance_max": dists_max,
            "area": areas})

def calc_barcode_fdr(b, cb):
    blanks = b[b.barcode_id.isin(cb.get_blank_indexes())]
    blanksNum = blanks.shape[0]
    totalNum = b.shape[0] + 1 # add psudo count
    fdr = (blanksNum / len(cb.get_blank_indexes())) / (totalNum / cb.get_barcode_count())
    return fdr

def estimate_lik_err_table(
    bd, cb, minScore=0, maxScore=10, bins=100):
    scores = np.linspace(minScore, maxScore, bins)
    blnkBarcodeNum = len(cb.get_blank_indexes())
    codeBarcodeNum = len(cb.get_coding_indexes()) + len(cb.get_blank_indexes())
    pvalues = dict()
    for s in scores:
        bd = bd[bd.likelihood >= s]
        numPos = np.count_nonzero(
            bd.barcode_id.isin(cb.get_coding_indexes()))
        numNeg = np.count_nonzero(
            bd.barcode_id.isin(cb.get_blank_indexes()))
        numNegPerBarcode = numNeg / blnkBarcodeNum
        numPosPerBarcode = (numPos + numNeg) / codeBarcodeNum
        pvalues[s] = numNegPerBarcode / numPosPerBarcode
    return pvalues

def estimate_barcode_threshold(
    barcodes,
    codebook,
    cutoff: float = 0.05,
    bins: int = 100):

    tab = estimate_lik_err_table(
        barcodes, 
        codebook, 
        minScore=0, 
        maxScore=10, 
        bins=bins)
    
    return min(np.array(list(tab.keys()))[
        np.array(list(tab.values())) <= cutoff])

def export_barcodes(
    obj,
    fnames: list = None):

    barcodes = []
    for fname in fnames:
        x = pd.read_hdf(fname, key="barcodes")
        x = x.assign(global_x = np.array(x.x * obj.get_microns_per_pixel()) +\
            np.array(obj.get_fov_offset(x.fov)[0]))
        x = x.assign(global_y = np.array(x.y * obj.get_microns_per_pixel()) +\
            np.array(obj.get_fov_offset(x.fov)[1]))
        x = x.assign(gene_name = \
            np.array(obj.get_codebook().get_data()["name"][
                x.barcode_id.astype(int)]))
        barcodes.append(x)

    # combine barcodes
    barcodes = pd.concat(
        barcodes, ignore_index = True)

    return barcodes
    
def filter_barcodes(barcodes,
                    codebook,
                    likelihoodThreshold: float = None,
                    keepBlankBarcodes: bool = False,
                    minAreaSize: int = 1):
    
    barcodes = barcodes[barcodes.likelihood >= likelihoodThreshold]
    barcodes = barcodes[barcodes.area >= minAreaSize]
    if not keepBlankBarcodes:
        barcodes = barcodes[
            barcodes.barcode_id.isin(codebook.get_coding_indexes())]
    return barcodes   
