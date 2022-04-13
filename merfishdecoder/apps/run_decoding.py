import os
import pickle
import pandas as pd
import numpy as np

from merfishdecoder.core import zplane
from merfishdecoder.util import utilities
from merfishdecoder.util import decoder

#def run_job(
#    dataSetName: str = None,
#    fov: int = None,
#    zpos: float = None,
#    decodingImagesName: str = None,
#    outputName: str = None,
#    maxCores: int = 5,
#    borderSize: int = 80,
#    magnitudeThreshold: float = 0.0,
#    distanceThreshold: float = 0.60):
#
#    """
#    MERFISH Decoding.
#
#    Args
#    ----
#    dataSetName: input dataset name.
#    
#    fov: field view number.
#
#    zpos: z position in uM.
#    
#    decodingImagesName: decoding image file name.
#    
#    outputName: output file name.
#    
#    maxCores: max number of processors for parallel computing. 
#    
#    borderSize: number of pixels to be removed from the decoding
#           images.
#
#    magnitudeThreshold: the min magnitudes for a pixel to be decoded.
#             Any pixel with magnitudes less than magnitudeThreshold 
#             will be filtered prior to the decoding.
#             
#    distanceThreshold: the maximum distance between an assigned pixel
#             and the nearest barcode. Pixels for which the nearest barcode
#             is greater than distanceThreshold are left unassigned.
#
#    """
#
#    utilities.print_checkpoint("Decode MERFISH images")
#    utilities.print_checkpoint("Start")
#    
#    # generate zplane object
#    zp = zplane.Zplane(dataSetName,
#                       fov=fov,
#                       zpos=zpos)
#
#    # create the folder
#    os.makedirs(os.path.dirname(outputName),
#                exist_ok=True)
#
#    # load readout images
#    f = np.load(decodingImagesName)
#    decodingImages = f[f.files[0]]
#    f.close()
#    
#    # pixel based decoding
#    decodedImages = decoder.decoding(
#             obj = zp,
#             movie = decodingImages,
#             borderSize = borderSize,
#             distanceThreshold = distanceThreshold,
#             magnitudeThreshold = magnitudeThreshold,
#             numCores = maxCores)
#    
#    # save decoded images
#    np.savez(outputName,
#             decodedImage=decodedImages["decodedImage"], 
#             magnitudeImage=decodedImages["magnitudeImage"],
#             distanceImage=decodedImages["distanceImage"])
#
#    utilities.print_checkpoint("Done")

def run_job(dataSetName: str = None,
            fov: int = None,
            zpos: float = None,
            barcodeWeightName: str = None,
            bitWeightName: str = None,
            decodingImagesName: str = None,
            outputName: str = None,
            maxCores: int = 5,
            borderSize: int = 80,
            minProb: float = 0.1,
            distanceThreshold: float = 0.2,
            magnitudeThreshold: float = 1.0,
            decodeMethod: str = "distance"):

    # print input variables
    print("====== input ======")
    print("dataSetName: %s" % dataSetName)
    print("fov: %d" % fov)
    print("zpos: %f" % zpos)
    print("barcodeWeightName: %s" % barcodeWeightName)
    print("bitWeightName: %s" % bitWeightName)
    print("decodingImagesName: %s" % decodingImagesName)
    print("outputName: %s" % outputName)
    print("borderSize: %d" % borderSize)
    print("minProb: %s" % minProb)
    print("distanceThreshold: %s" % distanceThreshold)
    print("magnitudeThreshold: %r" % magnitudeThreshold)
    print("decodeMethod: %r" % decodeMethod)
    print("==================\n")

    utilities.print_checkpoint("Decode MERFISH images")
    utilities.print_checkpoint("Start")
    
    # generate zplane object
    zp = zplane.Zplane(dataSetName,
                       fov=fov,
                       zpos=zpos)
    cb = zp.get_codebook()
    
    # load barcode weight + pseudocount
    if barcodeWeightName is not None:
        barcodeWeight = np.load(barcodeWeightName).astype(np.float)
    else:
        barcodeWeight = np.ones(cb.get_barcode_count()) / cb.get_barcode_count()

    if bitWeightName is not None:
        bitWeight = np.load(bitWeightName).astype(np.float)
    else:
        bitWeight = np.ones(cb.get_bit_count()) / cb.get_bit_count()
        
    # get codebook
    cbMat = cb.get_barcodes()

    # create the folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)
    
    # load readout images
    decodingImages = np.load(decodingImagesName)
    decodingImages = decodingImages["arr_0"]
    
    # filter pixels with probaility less than minProb
    if decodeMethod != "distance":
        for i in range(decodingImages.shape[0]):
            decodingImages[i][decodingImages[i] <= minProb] = 0
    
    decodedImages = decoder.decoding(obj = zp,
                                     movie = decodingImages,
                                     borderSize = borderSize,
                                     distanceThreshold = distanceThreshold,
                                     magnitudeThreshold = magnitudeThreshold,
                                     numCores = 1,
                                     barcodeWeight = barcodeWeight,
                                     bitWeight = bitWeight,
                                     decodeMethod = decodeMethod)

    # save decoded images
    np.savez(file = outputName,
             decodedImage=decodedImages["decodedImage"], 
             magnitudeImage=decodedImages["magnitudeImage"], 
             distanceImage=decodedImages["distanceImage"], 
             probabilityImage=decodedImages["probabilityImage"])

    utilities.print_checkpoint("Done")

def main():
    dataSetName = "191010_LMN7_DIV18_Map2Tau"
    barcodeWeightName = "barcodeScaleFactor.npy"
    bitWeightName = None
    fov = 188
    zpos = 0.0
    decodingImagesName = \
        "probImages/fov_{fov:d}_zpos_{zpos:.1f}.npz".format(
            fov = fov, zpos = zpos)
    outputName = \
        "decodedImages/fov_{fov:d}_zpos_{zpos:.1f}.jp.npz".format(
            fov = fov, zpos = zpos)
    maxCores = 1
    borderSize = 100
    minProb = 0.1
    magnitudeThreshold = 2.0
    distanceThreshold = 0.65
    decodeMethod = "joint_prob"
    
    run_job(
        dataSetName = dataSetName,
        fov = fov,
        zpos = zpos,
        barcodeWeightName = barcodeWeightName,
        decodingImagesName = decodingImagesName,
        outputName = outputName,
        minProb = minProb,
        maxCores = maxCores,
        borderSize = borderSize,
        magnitudeThreshold = magnitudeThreshold,
        distanceThreshold = distanceThreshold,
        decodeMethod = decodeMethod)

if __name__ == "__main__":
    main()

