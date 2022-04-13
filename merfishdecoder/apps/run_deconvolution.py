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
from merfishdecoder.util import deconvolution

def run_job(dataSetName: str = None,
            fov: int = None,
            zpos: float = None,
            probImagesName: str = None,
            deconvSigma: float = 1,
            deconvIterationCount: int = 10,
            outputName: str = None):

    # print input variables
    print("====== input ======")
    print("dataSetName: %s" % dataSetName)
    print("fov: %d" % fov)
    print("zpos: %f" % zpos)
    print("probImagesName: %s" % probImagesName)
    print("outputName: %s" % outputName)
    print("deconvSigma: %s" % deconvSigma)
    print("deconvIterationCount: %s" % deconvIterationCount)    
    print("==================\n")
    
    filterSize = int(2 * np.ceil(2 * deconvSigma + 1))

    # check points
    utilities.print_checkpoint("Process MERFISH images")
    utilities.print_checkpoint("Start")
    
    # create a zplane object
    zp = zplane.Zplane(dataSetName = dataSetName,
                       fov = fov,
                       zpos = zpos)
    
    # create the folder
    dirPath = os.path.dirname(outputName)
    os.makedirs(dirPath, exist_ok=True)

    # load readout images
    zp.load_processed_images(
        probImagesName)
    
    # run deconvolution
    points = []
    for bitName in zp.get_bit_name():
        bitImage = zp.get_readout_image_from_readout_name(bitName).copy()
        bitImageDeconv = deconvolution.deconvolve_lucyrichardson_guo(
            image = bitImage.astype(np.float),
            sigmaG = deconvSigma,
            windowSize = filterSize,
            iterationCount = deconvIterationCount)
        (rows, cols) = np.where(bitImageDeconv > 1)
        points.append(
            pd.DataFrame({"x": rows, 
                          "y": cols, 
                          "z": zp.get_bit_name().index(bitName),
                          "bitName": bitName,
                          "prob": bitImage[rows, cols],
                          "score": bitImageDeconv[rows, cols],
                      }))
    points = pd.concat(points, axis=0)
    points.to_hdf(outputName, key="dots", index=False)

    # write down the move
    utilities.print_checkpoint("Done")

def main():
    dataSetName = "191010_LMN7_DIV18_Map2Tau"
    fov = 188
    zpos = 5.0
    probImagesName = "probImages/fov_188_zpos_5.0.npz"
    outputName = "deconvImages/fov_188_zpos_5.0.h5"
    deconvSigma = 1.5
    deconvIterationCount = 10
    
    run_job(dataSetName = dataSetName,
            fov = fov,
            zpos = zpos,
            probImagesName = probImagesName,
            deconvSigma = deconvSigma,
            deconvIterationCount = deconvIterationCount,
            outputName = outputName)
    
if __name__ == "__main__":
    main()
