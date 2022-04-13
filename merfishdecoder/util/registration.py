import random
import numpy as np
from skimage import feature, registration, transform
from skimage.registration import phase_cross_correlation

import merfishdecoder
from merfishdecoder.core import zplane
from merfishdecoder.util import imagefilter
from merfishdecoder.util import utilities

def correct_drift(obj, 
                  frameNames: list = None,
                  refFrameIndex: int = 0,
                  highPassSigma: int = 3):
    
    """
    Correct mechanical drift using fiducial images.

    Args: 
        obj: a Zplane object
                             
        frameNames: a list of read names to perform correction.

        profile: a dictionary object that contains chromatic 
                  aberration correction profile.

        refFrameIndex: index of the frame used as referenece
                  image for calculating the drift offset.

    Returns: 
        A Zplane object
    """
    
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    
    if obj.get_readout_image_from_readout_name(
            frameNames[refFrameIndex]) is None:
        obj.load_readout_images(
            readoutNames = frameNames)

    if obj.get_fiducial_image_from_readout_name(
            frameNames[refFrameIndex]) is None:
        obj.load_fiducial_images(
            readoutNames = frameNames)
    
    obj = imagefilter.high_pass_filter(obj,
        frameNames = frameNames,
        readoutImage = False,
        fiducialImage = True,
        sigma = highPassSigma)

    fiducials = obj.get_fiducial_images(
        frameNames);
    
    # align all images to the ref image;
    random.seed(1);
    pcc = dict(zip(frameNames, [
        phase_cross_correlation(
            fiducials[refFrameIndex], 
            x, upsample_factor = 100) \
            for x in fiducials]))
    
    offsets = dict([ [k, x[0]] for k, x in pcc.items() ])
    errors = dict([ (k, x[1]) for k, x in pcc.items() ])
    
    transformations = dict([ (k, transform.SimilarityTransform(
        translation=[ -x[1], -x[0]])) for k, x in offsets.items() ])
    
    for fn in frameNames:
        obj._frames[fn]._fiducial = transform.warp(
                obj._frames[fn]._fiducial, 
                transformations[fn], 
                preserve_range=True
                ).astype(np.uint16)
        obj._frames[fn]._img = transform.warp(
                obj._frames[fn]._img, 
                transformations[fn], 
                preserve_range=True
                ).astype(np.uint16); 
    return (obj, errors)

def correct_chromatic_aberration(obj, 
                                 frameNames: list = None,
                                 profile: dict = None):

    """
    Correct the chromatic aberration.

    Args: 
        frameNames: A list of read names to perform correction.
            If frameNames is None (default), all the frames will 
            be corrected.

        profile: A dictionary object that contains chromatic 
            aberration correction profile. The profile was
            generated prior to the experiment.
                
    Returns: 
        A Zplane object.
    """
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    for fn in frameNames:
        frameColor = obj.get_image_color([fn])[0]
        if frameColor in profile:
            obj._frames[fn]._img = transform.warp(
                obj.get_readout_image_from_readout_name(fn), 
                profile[frameColor], 
                preserve_range=True
                ).astype("uint16");
        else:
            utilities.print_warning(
                "%s does not have color profile\n" % fn)
    return obj

