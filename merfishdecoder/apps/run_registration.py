import os
import pickle
import pandas as pd

import merfishdecoder
from merfishdecoder.core import dataset
from merfishdecoder.core import zplane
from merfishdecoder.util import registration
from merfishdecoder.util import utilities

def run_job(
    dataSetName: str,
    fov: int,
    zpos: float,
    outputName: str,
    registerDrift: bool = True,
    refFrameIndex: int=0,
    highPassFilterSigma: int=3,
    registerColor: bool=True,
    registerColorProfile: str = None,
    saveFiducials: bool=False):

    """
    Reorganization and registration of MERFISH images.

    Args
    ----
    dataSetName: input dataset name.

    fov: the field of view to be processed. 

    zpos: the z position of the selected FOV to be processed. Each 
                z-plane is preprocessed indepedently. 
    
    outputName: outputName file name.
    
    registerDrift: a boolen variable indicates whether images are registered
                for mechanical drift.
    
    refFrameIndex: the index of the frame  that serves as reference for 
                correcting chromatic abberation, mechanical drift. 
                Default is the first bit frame is used as reference [0].
    
    highPassFilterSigma: the size of the gaussian sigma used in the high pass 
                filter for removing the background prior to the warping.
    
    registerColor: a boolen variable indicates whether chromatic abberation 
                (CA) between different color channels is corrected. The 
                distortion caused by different color is corrected using 
                existing profile (colorProfileFileName) calibrated for the 
                microscopy.
    
    registerColorProfile: a pkl file that contains the chromatic aberration
                profile. If it is None, the default chromatic aberration
                profile will be used.
    
    saveFiducials: a boolen variable indicates whether aligned fiducial
                images are saved. Fiducial images will be saved
                in the same folder with the same prefix.
    """
    
    utilities.print_checkpoint("Register MERFISH images")
    utilities.print_checkpoint("Start")
    
    # generate zplane object
    zp = zplane.Zplane(dataSetName,
                       fov = fov,
                       zpos = zpos)

    # create the folder
    os.makedirs(os.path.dirname(outputName), 
                exist_ok=True)

    # load readout images
    zp.load_readout_images(zp.get_readout_name())
    
    # correct mechanical drift
    if registerDrift:
        (zp, errors) = registration.correct_drift(
            obj = zp,
            refFrameIndex = refFrameIndex,
            highPassSigma = highPassFilterSigma)
    
    # correct chromatic abberation
    if registerColor:
        if registerColorProfile is not None:
            profile = pickle.load(registerColorProfile)
        else:
            profile = zp.get_chromatic_aberration_profile()
        zp = registration.correct_chromatic_aberration(
            obj = zp,
            profile = profile)

    # save readout images
    zp.save_readout_images(
        fileName = outputName)

    # save the error
    errors = pd.DataFrame(errors.items())
    errors.columns = ["frameName", "error"]
    prefix = os.path.splitext(outputName)[0]
    errors.to_csv(
        prefix + "_err.csv",
        header = True, 
        index = False) 
    
    # save fiducial images
    saveFiducials and zp.save_fiducial_images(
        fileName = prefix + "_fiducial.tif")
    utilities.print_checkpoint("Done")

