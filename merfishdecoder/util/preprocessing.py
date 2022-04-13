import cv2
import numpy as np

from merfishdecoder.core import zplane


def log_readout_images(obj: zplane.Zplane = None, 
                       frameNames: list = None
                       ) -> zplane.Zplane:

    """

    Correct the intensity difference between color channels
               using existed scale factor profiles

    """
    
    frameNames = obj.get_readout_name() \
        if frameNames is None else frameNames
    for fn in frameNames:
        obj._frames[fn]._img = np.log10(obj._frames[fn]._img.astype(np.float16) + 1)
    return obj

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

"""Position-independent normalization
"""

from typing import Tuple, List
from scipy import stats
import numpy as np


class pin(object):
    def __init__(
        self,
        mu_1=1,
        sigma2_1=1,
        mu_blk=0,
        sigma2_blk=1,
        pie=0.1,
        tpr=0.9,
        tnr=0.8,
        p=0.8,
        mu_0=0.3,
        sigma2_0=1,
    ):
        """Init position-independent normalization method.

        - mu_1, sigma2_1: mean and variance for
          the log10-level light intensity when there is signal
        - mu_blk, sigma2_blk: mean and variance for
          the log10-level light intensity for the blank pixels
          when they show signals by chance.
        - pie: prior P(X = 1)
        - tpr: true positive rate
        - tnr: true negative rate
        - p: zero weight in hurdle model
        - mu_0, sigma2_0: truncated normal mean and variance in hurdle model
        """

        self.mu_1: float = mu_1
        self.sigma2_1: float = sigma2_1
        self.norm_1 = stats.norm(loc=self.mu_1, scale=np.sqrt(self.sigma2_1))

        self.mu_blk: float = mu_blk
        self.sigma2_blk: float = sigma2_blk
        self.norm_blk = stats.norm(loc=self.mu_blk, scale=np.sqrt(self.sigma2_blk))

        self.pie: float = pie
        self.tpr: float = tpr
        self.tnr: float = tnr

        ## hurdle model: non-zero part uses truncated normal
        self.p: float = p
        self.mu_0: float = mu_0
        self.sigma2_0: float = sigma2_0
        ## set a as 1e-15 to make sure the trunnorm.pdf(0) equals to 0
        ## then make a transformation since in stats.truncnorm, a and b are relatively
        ## towards a standard norm
        a = (1e-15 - self.mu_0) / np.sqrt(self.sigma2_0)
        # b = stats.norm.ppf(0.999, loc=self.mu_0, scale=np.sqrt(self.sigma2_0))
        # b = (b - self.mu_0) / np.sqrt(self.sigma2_0)
        b = np.inf
        self.truncnorm = stats.truncnorm(
            a=a,
            b=b,
            loc=self.mu_0,
            scale=np.sqrt(self.sigma2_0),
        )

    def _prob(self, y_ij: float) -> float:
        """Calculate P(X_ij | y_ij)

        Return:
        - P(X_ij = 1 | y_ij)
        """
        p_yij_1 = self.norm_1.pdf(y_ij)
        p_yij_blank = self.norm_blk.pdf(y_ij)

        w_xij_1: float = self.pie * (self.tpr * p_yij_1 + (1 - self.tpr) * p_yij_blank)
        w_xij_0: float = (1 - self.pie) * (
            (1 - self.tnr) * p_yij_1 + self.tnr * p_yij_blank
        )
        p = w_xij_1 / (w_xij_1 + w_xij_0)
        return p

    def prob1_vec(self, y_t: np.ndarray) -> np.ndarray:
        """Given a matrix (figure at t-th turn), get the P(X^t = 1 | Y^t).
        This function is to accelarate the process through vectorization.
        """
        ## P(x = 1) * P(y|x)
        p_y_1: np.ndarray = self.norm_1.pdf(y_t)
        p_y_blk: np.ndarray = self.norm_blk.pdf(y_t)
        w_x_1: np.ndarray = self.pie * (self.tpr * p_y_1 + (1 - self.tpr) * p_y_blk)

        ## P(x = 0) * P(y|x)
        p_y_0: np.ndarray = self.truncnorm.pdf(y_t)
        h_y_0: np.ndarray = (1 - self.p) * p_y_0
        h_y_0[y_t == 0] = self.p
        w_x_0: np.ndarray = (1 - self.pie) * (
            self.tnr * h_y_0 + (1 - self.tnr) * p_y_blk
        )
        ## point-wise divide
        r = np.divide(w_x_1, w_x_0 + w_x_1)
        return r

    def prob1(self, y_t: np.ndarray) -> np.ndarray:
        """Given a matrix (figure at t-th turn), get the P(X^t = 1 | Y^t)."""
        ## vectorize too slow
        # r = np.vectorize(self._prob)(y_t)
        r = np.asarray(np.frompyfunc(self._prob, 1, 1)(y_t), dtype=np.float)
        return r
