import pandas as pd
import numpy as np
import gc
from numpy import linalg as LA
import random
import cv2
import copy
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
import SharedArray as sa
from skimage import measure
import tempfile
import os
from collections import Counter
from numba import jit
import gc
from scipy import stats, special

from merfishdecoder.data import codebook as cb
from merfishdecoder.util import utilities
from merfishdecoder.core import zplane

def decoding(obj: zplane.Zplane = None,
             movie: np.ndarray = None,
             borderSize: int = 100,
             distanceThreshold: float = 0.65,
             magnitudeThreshold: float = 0.0,
             numCores: int = 1,
             barcodeWeight: np.ndarray = None,
             bitWeight: np.ndarray = None,
             decodeMethod: str = "distance"):

    """
    Pixel-based decoding.

    Args
    ----
    dataSetName: input dataset name.
    
    decodingMovieFile: movie for decoding.

    numCores: number of processors for parallel computing. 
             
    magnitudeThreshold: the min magnitudes for a pixel to be decoded.
             Any pixel with magnitudes less than magnitudeThreshold 
             will be filtered prior to the decoding.
             
    distanceThreshold: the maximum distance between an assigned pixel
             and the nearest barcode. Pixels for which the nearest barcode
             is greater than distanceThreshold are left unassigned.
    
    numCores: number of threads for decoding
    """

    # remove the bordering pixels of decoding movie
    imageSize = movie.shape[1:]
    borderImage = np.ones(imageSize)
    borderImage[
        borderSize:(imageSize[0] - borderSize),
        borderSize:(imageSize[0] - borderSize)] = 0
    movie = np.array([ x * (1 - borderImage) for x in movie ])
    
    # pixel-based decoding
    if decodeMethod == "distance":
        decodeDict = pixel_based_decode_distance(
            movie = movie,
            codebookMat = obj.get_codebook().get_barcodes().astype(np.float32),
            distanceThreshold = distanceThreshold,
            magnitudeThreshold = magnitudeThreshold,
            oneBitThreshold = 1,
            numCores = numCores)
    elif decodeMethod == "cross_entropy":
        decodeDict = pixel_based_decode_cross_entropy(
             movie = movie,
             codebookMat = obj.get_codebook().get_barcodes().astype(np.float32),
             bitWeight = bitWeight,
             magnitudeThreshold = magnitudeThreshold)
    elif decodeMethod == "joint_prob":
        decodeDict = pixel_based_decode_joint_prob(
            movie  = movie,
            codebookMat = obj.get_codebook().get_barcodes().astype(np.float32),
            barcodeWeight = barcodeWeight,
            magnitudeThreshold = magnitudeThreshold)

    return decodeDict

def pixel_based_decode_joint_prob(
    movie: np.ndarray,
    codebookMat: np.ndarray,
    barcodeWeight: np.ndarray,
    magnitudeThreshold: float = 1.0
    ) -> dict:
    
    magnitudeImage = movie.sum(axis=0)
    rows, cols = np.asarray(
        magnitudeImage >= magnitudeThreshold).nonzero()
    
    r = codebookMat.copy()
    w = barcodeWeight.copy()
    
    w = np.insert(w, obj=0, values=np.median(w), axis=0)
    r = np.insert(r, obj=0, values=np.zeros(r.shape[1]), axis=0)
    
    eps = 1e-15
    y_1 = np.empty(shape=(r.shape[1], rows.shape[0]), dtype=np.float)
    y_0 = np.empty_like(y_1, dtype=np.float)
    for t in range(movie.shape[0]):
    	t1 = movie[t]
    	t2 = 1 - t1
    	t1[t1 == 0] = eps
    	t2[t2 == 0] = eps
    	y_1[t] = np.log(t1[rows, cols])
    	y_0[t] = np.log(t2[rows, cols])
    ll = np.matmul(r, y_1) + np.matmul(1 - r, y_0)
    logw = np.log(w / w.sum())
    logp = np.repeat(logw, ll.shape[1]).reshape(ll.shape) + ll
    prob = np.exp(logp - special.logsumexp(logp, axis=0))
    assigned_rna = np.argmax(prob, axis=0) - 1
    assigned_rna_p = np.max(prob, axis=0)
    
    decodedImage = -np.ones(movie.shape[1:])
    probabilityImage = np.zeros(movie.shape[1:])

    decodedImage[rows, cols] = assigned_rna
    probabilityImage[rows, cols] = assigned_rna_p

    return dict({
        "decodedImage": decodedImage,
        "magnitudeImage": magnitudeImage,
        "distanceImage": probabilityImage,
        "probabilityImage": probabilityImage})

def pixel_based_decode_cross_entropy(
    movie: np.ndarray,
    codebookMat: np.ndarray,
    bitWeight: np.ndarray,
    magnitudeThreshold: float = 1.0
    ) -> dict:

    eps = 1e-15
    movie[movie == 1] = 1 - eps
    movie[movie == 0] = eps
    
    magnitudeImage = movie.sum(axis=0)
    rows, cols = np.asarray(
        magnitudeImage >= magnitudeThreshold).nonzero()
        
    ## 48 by col_num
    y_pin_s2d = movie[:, rows, cols]
    log_p = np.log(y_pin_s2d)
    log_1_p = np.log(1 - y_pin_s2d)
    ce = -(np.matmul(codebookMat, log_p) + np.matmul(1 - codebookMat, log_1_p))
    
    assigned_rna = np.argmin(ce, axis = 0)
    assigned_rna_ce = np.min(ce, axis=0)
    
    decodedImage = - np.ones(movie.shape[1:])
    probabilityImage = np.zeros(movie.shape[1:])
    decodedImage[rows, cols] = assigned_rna
    probabilityImage[rows, cols] = assigned_rna_ce

    return dict({
        "decodedImage": decodedImage,
        "magnitudeImage": magnitudeImage,
        "distanceImage": probabilityImage,
        "probabilityImage": probabilityImage})
    
def pixel_based_decode_distance(
    movie: np.ndarray,
    codebookMat: np.ndarray,
    numCores: int = 1,
    distanceThreshold: float = 0.65,
    magnitudeThreshold: float = 0,
    oneBitThreshold: int = 1
    ) -> dict:
  
    """
    NOTE: THIS FUNCTION IS COPIED/MODIFIED FROM MERLIN
    https://github.com/emanuega/MERlin
    
    Purpose:
           Assign barcodes to the pixels in the provided image stock.
           Each pixel is assigned to the nearest barcode from the codebook if
           the distance between the normalized pixel trace and the barcode is
           less than the distance threshold.
    Args:
        movie: input image stack. The first dimension indexes the bit
            number and the second and third dimensions contain the
            corresponding image.
           
        codebook: a codebook object that contains the barcode for MERFISH.
            A valid codebook should contains both gene and blank batcodes.
        
        distanceThreshold: the maximum distance between an assigned pixel
            and the nearest barcode. Pixels for which the nearest barcode
            is greater than distanceThreshold are left unassigned.

        magnitudeThreshold: the minimum magnitude for a pixel. Pixels for 
            which magnitude is smaller than magnitudeThreshold are left unassigned.
            Note that the magnitude is scaled by the median value of each bit.

        numCores: number of processors used for decoding
        
    Returns:
        A dictionary object contains the following images:
            decodedImage - assigned barcode id
            magnitudeImage - magnitude for the pixel
            distanceImage - min dsitance to the closest barcode
            
    """
    
    bitNum = codebookMat.sum(axis=1)[0]
    codebookMatWeighted = codebookMat[:,:] / \
        cal_pixel_magnitude(codebookMat)[:, None]
    
    imageSize = movie.shape[1:]
    
    pixelTraces = np.reshape(
        movie, 
        (movie.shape[0], 
        np.prod(movie.shape[1:]))).T
    numPixel = pixelTraces.shape[0]
    
    pixelMagnitudes = cal_pixel_magnitude(
        pixelTraces.astype(np.float32))
    
    pixelTracesCov = np.array(
        [np.count_nonzero(x) for x in pixelTraces])

    pixelIndexes = np.where(
        (pixelTracesCov >= oneBitThreshold) &
        (pixelMagnitudes >= magnitudeThreshold))[0]
    
    if pixelIndexes.shape[0] == 0:
        return dict({
            "decodedImage": np.empty(0),
            "magnitudeImage": np.empty(0),
            "distanceImage": np.empty(0),
            "probabilityImage": np.empty(0)
        })

    pixelTraces = pixelTraces[pixelIndexes]
    pixelMagnitudes = pixelMagnitudes[pixelIndexes]
    
    normalizedPixelTraces = \
        pixelTraces / pixelMagnitudes[:, None]
    
    del pixelTraces
    del pixelTracesCov
    gc.collect()
    
    neighbors = NearestNeighbors(
        n_neighbors = 1, 
        algorithm = 'ball_tree')
    
    neighbors.fit(codebookMatWeighted)
    
    if numCores > 1:
        normalizedPixelTracesSplits = np.array_split(
            normalizedPixelTraces, 100)
        with mp.Pool(processes=numCores) as pool:
            results = pool.starmap(kneighbors_func, 
                zip(normalizedPixelTracesSplits, [neighbors] * 100))
        distances = np.vstack([ x[0] for x in results ])
        indexes = np.vstack([ x[1] for x in results ])
    else:
        results = kneighbors_func(
            normalizedPixelTraces, neighbors)
        distances = results[0]
        indexes = results[1]

    del normalizedPixelTraces
    gc.collect()
    pixelTracesDecoded = -np.ones(
        numPixel, 
        dtype=np.int16)

    pixelTracesDecoded[pixelIndexes] = \
        np.array([i if ( d <= distanceThreshold ) else -1
              for i, d in zip(indexes, distances)], 
        dtype=np.int16)
    
    decodedImage = np.reshape(
       pixelTracesDecoded,
       imageSize)
    
    pixelDistanceTraces = np.ones(
        numPixel, 
        dtype=np.float32
        ) * distanceThreshold
    
    pixelDistanceTraces[pixelIndexes] = \
        np.ravel(distances)

    pixelDistanceTraces = np.clip(
        a=pixelDistanceTraces, 
        a_min=0, 
        a_max=distanceThreshold)
    
    distanceImage = np.reshape(
       pixelDistanceTraces,
       imageSize)
        
    pixelMagnitudeTraces = np.zeros(
        numPixel, 
        dtype=np.float32)

    pixelMagnitudeTraces[pixelIndexes] = \
        np.ravel(pixelMagnitudes)
    
    magnitudeImage = np.reshape(
       pixelMagnitudeTraces,
       imageSize)

    return dict({
        "decodedImage": decodedImage,
        "magnitudeImage": magnitudeImage,
        "distanceImage": distanceImage,
        "probabilityImage": distanceImage })

def calc_pixel_probability(
    model,
    decodedImage:   np.ndarray = None,
    magnitudeImage: np.ndarray = None,
    distanceImage:  np.ndarray = None,
    minProbability: float = 0.01):
    
    m = np.log10(magnitudeImage[decodedImage > -1])
    d = distanceImage[decodedImage > -1]
    p = model.predict_proba(
        np.array([m, d]).T)[:,1]
    probabilityImage = np.zeros(decodedImage.shape) + minProbability
    probabilityImage[decodedImage > -1] = p
    return probabilityImage

@jit(nopython=True)
def cal_pixel_magnitude(x):

    """
    Calculate magnitude for pixel trace x 
    """

    pixelMagnitudes = np.array([ np.linalg.norm(x[i]) \
        for i in range(x.shape[0]) ], dtype=np.float32)
    pixelMagnitudes[pixelMagnitudes == 0] = 1 
    return(pixelMagnitudes)  

def kneighbors_func(x, n):
    return(n.kneighbors(
        x, return_distance=True))
