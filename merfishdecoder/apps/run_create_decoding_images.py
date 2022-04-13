import os
import pickle
import pandas as pd
import numpy as np
import cv2
from scipy import stats, special
from scipy import stats, special, signal

from merfishdecoder.core import zplane
from merfishdecoder.util import utilities
from merfishdecoder.util import decoder

def run_job(
    dataSetName: str = None,
    fov: int = None,
    zpos: float = None,
    decodingImagesName: str = None,
    outputName: str = None,
    maxCores: int = 5,
    borderSize: int = 80,
    magnitudeThreshold: float = 0.0,
    distanceThreshold: float = 0.60):

    utilities.print_checkpoint("Decode MERFISH images")
    utilities.print_checkpoint("Start")
    
    # generate zplane object
    zp = zplane.Zplane(dataSetName,
                       fov=fov,
                       zpos=zpos)

    # create the folder
    os.makedirs(os.path.dirname(outputName),
                exist_ok=True)
    
    upFactor = 2
    # load readout images
    predictedPositionsName = "predictedPositions/fov_188_zpos_5.0.h5"
    df = pd.read_hdf(predictedPositionsName)
    df = df[df.p >= 0.5]

    # create decoding images
    decodingImages = np.zeros((
        zp.get_bit_count(), 
        zp.get_image_size()[0] * upFactor, 
        zp.get_image_size()[1] * upFactor))

    for bitName in zp.get_bit_name():
        df_i = df[df.bitName == bitName]
        pixelSize = zp.get_microns_per_pixel() * 1000 / upFactor
        decodingImages[
            zp.get_bit_name().index(bitName), 
            (df_i.x / pixelSize).astype(np.int), 
            (df_i.y / pixelSize).astype(np.int)
        ] = df_i.p
    
    
    # add low pass filter
    filteredImages = np.zeros(decodingImages.shape, dtype=np.float)
    if kernelSize > 0:
        mask = np.ones((kernelSize,kernelSize))
        for i in range(zp.get_bit_count()):
            filteredImages[i,:,:] = signal.convolve2d(decodingImages[i,:,:].astype(np.float), mask, mode='same')
    else:
        for i in range(zp.get_bit_count()):
            filteredImages[i,:,:] = decodingImages[i,:,:].astype(np.float)
    filteredImages[filteredImages > 1] = 1
    np.savez(outputName, filteredImages)
    utilities.print_checkpoint("Done")

def main():
    dataSetName = "191010_LMN7_DIV18_Map2Tau"
    fov = 188
    zpos = 5.0
    predictedPositionsName = "predictedPositions/fov_188_zpos_5.0.h5"
    outputName = "decodingImages/fov_188_zpos_5.0.npz"
    maxCores = 5
    borderSize = 80
    
    run_job(
        dataSetName = dataSetName,
        fov = fov,
        zpos = zpos,
        decodingImagesName = decodingImagesName,
        outputName = outputName,
        maxCores = maxCores,
        borderSize = borderSize,
        magnitudeThreshold = magnitudeThreshold,
        distanceThreshold = distanceThreshold)
    
if __name__ == "__main__":
    main()


