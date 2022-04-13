import os
import pickle
import tifffile
import pandas as pd
import numpy as np
from scipy import stats, special, signal

from merfishdecoder.core import zplane
from merfishdecoder.util import imagefilter
from merfishdecoder.util import preprocessing
from merfishdecoder.util import utilities

def run_job(dataSetName: str = None,
            fov: int = None,
            zpos: float = None,
            processedImagesName: str = None,
            outputName: str = None,
            modelName: str = None,
            kernelSize: int = 3):

    # print input variables
    print("====== input ======")
    print("dataSetName: %s" % dataSetName)
    print("fov: %d" % fov)
    print("zpos: %f" % zpos)
    print("processedImagesName: %s" % processedImagesName)
    print("outputName: %s" % outputName)
    print("modelName: %s" % modelName)
    print("kernelSize: %s" % kernelSize)    
    print("==================\n")

    # check points
    utilities.print_checkpoint("Process MERFISH images")
    utilities.print_checkpoint("Start")

    # create a zplane object
    zp = zplane.Zplane(dataSetName,
                       fov = fov,
                       zpos = zpos)
    
    # load codebook matrix
    cbookMatrix = zp.get_codebook().get_barcodes()
    
    # create the folder
    dirPath = os.path.dirname(outputName)
    os.makedirs(
        dirPath,
        exist_ok=True)

    # load readout images
    zp.load_processed_images(
        processedImagesName)
    
    # log10 transform
    zp = preprocessing.log_readout_images(
        obj = zp,
        frameNames = zp.get_bit_name())
    
    # load the model
    modelDict = pickle.load(
        open(modelName, "rb"))

    probImageList = []
    for bitName in zp.get_bit_name():
        model = modelDict[bitName]
        bitImage = zp.get_readout_image_from_readout_name(bitName).copy()
        prob = modelDict[bitName].predict_proba(bitImage.reshape(-1, 1))
        probImage = prob[:,1].reshape(zp.get_image_size())
        probImageList.append(probImage)
    probImages = np.array(probImageList) 
    
    if kernelSize > 0:
        mask = np.ones((kernelSize,kernelSize))
        for i in range(zp.get_bit_count()):
            grad = signal.convolve2d(probImages[i].astype(np.float), mask, mode='same')
            grad = grad / grad.max()
            probImages[i] = grad * probImages[i]

    # write down the move
    np.savez_compressed(
            outputName,
            probImages.astype(np.float))

    utilities.print_checkpoint("Done")

def main():
    dataSetName = "191010_LMN7_DIV18_Map2Tau"
    fov = 188
    zpos = 0.0
    processedImagesName = "processedImages/fov_188_zpos_0.0.npz"
    outputName = "probImages/fov_188_zpos_0.0.npz"
    modelName = "gmModelParam.v2.pkl"
    kernelSize = 3
    
    run_job(dataSetName = dataSetName,
            fov = fov,
            zpos = zpos,
            processedImagesName = processedImagesName,
            outputName = outputName,
            modelName = modelName,
            kernelSize= kernelSize)

if __name__ == "__main__":
    main()
