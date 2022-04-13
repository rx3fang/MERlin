import cv2
import numpy as np

from merfishdecoder.core import zplane


def low_pass_filter(obj: zplane.Zplane = None, 
                         frameNames: list = None,
                         sigma = 1,
                         windowSize = 3
                         ) -> zplane.Zplane:
    """
    Correct the intensity difference between color channels
               using existed scale factor profiles
    """
    
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    for fn in frameNames:
        obj._frames[fn]._img = cv2.GaussianBlur(
            obj._frames[fn]._img.astype(np.float32),
            (windowSize, windowSize),
            sigma,
            borderType = cv2.BORDER_REPLICATE)
    return obj

def high_pass_filter(obj: zplane.Zplane = None,
                     frameNames: list = None,
                     readoutImage: bool = True,
                     fiducialImage: bool = False,
                     sigma: int = 3
                     ) -> zplane.Zplane:

    """
    Purpuse: 
        High pass filter to remove the cell background

    Args: 
        obj: a zplane object
        
        frameNames: a list of read names to perform correction.
        
        readoutImage: a boolen variable indicates whether to apply
                high pass filter to readout image.

        fiducialImage: a boolen variable indicates whether to apply
                high pass filter to fiducial image.

        sigma: low pass sigma for high pass filter.

    Returns: 
        N/A
    """
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    windowSize = int(2 * np.ceil(2 * sigma) + 1)
    if readoutImage:
        for fn in frameNames:
            obj._frames[fn]._img = \
                _high_pass_filter(
                    image = obj._frames[fn]._img, 
                    windowSize = windowSize,
                    sigma = sigma)
    if fiducialImage:
        for fn in frameNames:
            obj._frames[fn]._fiducial = \
                _high_pass_filter(
                    image = obj._frames[fn]._fiducial, 
                    windowSize = windowSize,
                    sigma = sigma)
    return obj

def _high_pass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float) -> np.ndarray:

    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """

    lowpass = cv2.GaussianBlur(image,
                               (windowSize, windowSize),
                               sigma,
                               borderType=cv2.BORDER_REPLICATE)
    gauss_highpass = image - lowpass
    gauss_highpass[lowpass > image] = 0
    return gauss_highpass

def scale_readout_images(obj: zplane.Zplane = None, 
                         frameNames: list = None,
                         scaleFactors: dict = None,
                         ) -> zplane.Zplane:

    """

    Correct the intensity difference between color channels
               using existed scale factor profiles

    """
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    if scaleFactors is None:
        scaleFactors = estimate_scale_factors(
            obj, frameNames)
    for fn in frameNames:
        obj._frames[fn]._img = obj._frames[fn]._img.astype(np.float16) / scaleFactors[fn]
    return obj

def estimate_scale_factors(obj: zplane.Zplane = None, 
                           frameNames: list = None
                           ) -> dict:

    """
    Estimate scale factors between rounds of images.
    """

    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames

    return dict(zip(frameNames,
        [ np.median(x[x > 0]) for x in 
        obj.get_readout_images(frameNames) ]))


