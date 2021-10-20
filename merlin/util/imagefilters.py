import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def high_pass_filter(image: np.ndarray,
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

def low_pass_filter(imageData: np.ndarray,
                     lowPassSigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        lowPassSigma: the sigma of the Gaussian.

    Returns:
        the low pass filtered image. The returned image is the same type
        as the input image.
    """
    
    filteredImages = np.zeros(imageData.shape, dtype=np.float32)
    filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
    for i in range(imageData.shape[0]):
        filteredImages[i, :, :] = cv2.GaussianBlur(
            imageData[i, :, :], (filterSize, filterSize), lowPassSigma)
    return filteredImages

    