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
        self.mu_0: float = mu_0
        self.sigma2_0: float = sigma2_0
        
        self.truncnorm = stats.norm(loc=self.mu_0, scale=np.sqrt(self.sigma2_0))

        self.pie: float = pie
        self.tpr: float = tpr
        self.tnr: float = tnr

        ## hurdle model: non-zero part uses truncated normal
        self.p: float = p
        self.mu_0: float = mu_0
        self.sigma2_0: float = sigma2_0
        ### set a as 1e-15 to make sure the trunnorm.pdf(0) equals to 0
        ### then make a transformation since in stats.truncnorm, a and b are relatively
        ### towards a standard norm
        #a = (1e-15 - self.mu_0) / np.sqrt(self.sigma2_0)
        ## b = stats.norm.ppf(0.999, loc=self.mu_0, scale=np.sqrt(self.sigma2_0))
        ## b = (b - self.mu_0) / np.sqrt(self.sigma2_0)
        #b = np.inf
        #self.truncnorm = stats.truncnorm(
        #    a=a,
        #    b=b,
        #    loc=self.mu_0,
        #    scale=np.sqrt(self.sigma2_0),
        #)

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


class pins(object):
    def __init__(
        self,
        mu_sigma2_1: np.ndarray,
        mu_sigma2_blk: Tuple[float, float],
        p_mu_sigma2_0: np.ndarray,
        pie: float = 0.1,
        tpr: float = 0.9,
        tnr: float = 0.8,
    ):
        self.mu_sigma2_1 = mu_sigma2_1
        self.mu_sigma2_blk = mu_sigma2_blk
        self.num_turns = self.mu_sigma2_1.shape[0]
        self.p_mu_sigma2_0 = p_mu_sigma2_0
        self.pie = pie
        self.tpr = tpr
        self.tnr = tnr
        self.__init_pin()

    def __init_pin(self) -> None:
        self.pins: List[pin] = []
        for i in range(self.num_turns):
            self.pins.append(
                pin(
                    mu_1=self.mu_sigma2_1[i, 0],
                    sigma2_1=self.mu_sigma2_1[i, 1],
                    mu_blk=self.mu_sigma2_blk[0],
                    sigma2_blk=self.mu_sigma2_blk[1],
                    pie=self.pie,
                    tpr=self.tpr,
                    tnr=self.tnr,
                    p=self.p_mu_sigma2_0[i, 0],
                    mu_0=self.p_mu_sigma2_0[i, 1],
                    sigma2_0=self.p_mu_sigma2_0[i, 2],
                )
            )

    def prob1(self, y: np.ndarray) -> np.ndarray:
        r = np.empty_like(y, dtype=np.float)
        for i in range(r.shape[0]):
            print(f"pin normalization on {i}th-turn ... ")
            r[i] = self.pins[i].prob1_vec(y[i])
        return r

"""Position-independent decoding.
"""

from typing import Tuple, List
from scipy import stats, special
import numpy as np

class pid(object):
    def __init__(
        self,
        w: np.ndarray,
        mu_sigma2_1: np.ndarray,
        mu_sigma2_blk: np.ndarray,
        p_mu_sigma2_0: np.ndarray,
        tpr: float = 0.9,
        tnr: float = 1.0,
    ):
        """Init position-independent decoding.
        - w: a vector of weight for different RNA species
        - mu_sigma2_1: num_turn by 2
          - mu_1, sigma2_1: mean and variance of
            the log10-level light intensity when there is signal
        - mu_sigma2_blk:
          - mu_blk, sigma2_blk: mean and variance for
            the log10-level light intensity for the blank pixels
            when they show signals by chance.
        - p_mu_sigma2_ab_0: num_turn by 5
          - p: zero weight in hurdle model
          - mu_0, sigma2_0: truncated normal mean and variance in hurdle model
        - tpr: true positive rate
        - tnr: true negative rate, default is 1.0
        """
        self.logw: np.ndarray = np.log(w)
        self.mu_sigma2_1: np.ndarray = mu_sigma2_1
        self.norm_1 = [stats.norm(loc=m, scale=np.sqrt(v)) for m, v in self.mu_sigma2_1]

        self.mu_blk: np.ndarray = mu_sigma2_blk[0]
        self.sigma2_blk: np.ndarray = mu_sigma2_blk[1]
        self.norm_blk = stats.norm(loc=self.mu_blk, scale=np.sqrt(self.sigma2_blk))

        self.tpr: float = tpr
        self.tnr: float = tnr

        ## hurdle model: non-zero part uses truncated normal
        self.p_mu_sigma2_0: np.ndarray = p_mu_sigma2_0
        self.ptruncnorm = [
            (
                p,
                stats.norm(
                    loc=m,
                    scale=np.sqrt(v),
                    #a=(0.0 - m) / np.sqrt(v),
                    #b=(1.0 - m) / np.sqrt(v),
                ),
            )
            for p, m, v in self.p_mu_sigma2_0
        ]

    def layerwise_likelihood(
        self, t: int, y_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given t and a matrix (figure at t-th turn), get two likelihoods:
           - p(Y^t | X^t = 1)
           - p(Y^y | X^t = 0)
        This function is to accelarate the process through vectorization.
        """
        ## P(y|x=1)
        p_y_1: np.ndarray = self.norm_1[t].pdf(y_t)
        p_y_blk: np.ndarray = self.norm_blk.pdf(y_t)
        w_x_1: np.ndarray = self.tpr * p_y_1 + (1 - self.tpr) * p_y_blk

        ## P(y|x=0)
        p, tn = self.ptruncnorm[t]
        p_y_0: np.ndarray = tn.pdf(y_t)
        h_y_0: np.ndarray = (1 - p) * p_y_0
        h_y_0[y_t < 1e-15] = p
        w_x_0: np.ndarray = self.tnr * h_y_0 + (1 - self.tnr) * p_y_blk

        return w_x_1, w_x_0

    def logpostr(
        self, y: np.ndarray, r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the posterior of X_ij on each RNA species.
        y: light intensities of MERFISH experiment, num_turn by row by col
           - has been log10-transformed
        r: MERFISH cookbook, num_RNA_species by num_turn
           - by default, we treat the first one as null

        Return:
        - log posterior probability tensor: num_RNA_species by row by col
        - two log likelihoods for each layer, num_turn by row by col
          - y_1: log p(Y | X = 1)
          - y_2: log p(Y | X = 0)
        """
        y_1 = np.empty_like(y, dtype=np.float16)
        y_0 = np.empty_like(y, dtype=np.float16)
        eps = 1e-15
        for t in range(y.shape[0]):
            print(f"pid laylerwise likelihood on {t}th-turn ... ")
            t1, t2 = self.layerwise_likelihood(t, y[t])
            t1[t1 == 0] = eps
            t2[t2 == 0] = eps
            y_1[t] = np.log(t1)
            y_0[t] = np.log(t2)
        print(f"pid join the likelihood ... ")
        ll = np.tensordot(r, y_1, axes=[1, 0]) + np.tensordot(1 - r, y_0, axes=[1, 0])
        ## add log of prior to each likelihood
        print(f"pid add the log prior ... ")
        logp = np.repeat(self.logw, ll.shape[1] * ll.shape[2]).reshape(ll.shape) + ll
        return logp, y_1, y_0

    def map(self, logp: np.ndarray, axis=0) -> Tuple[np.ndarray, np.ndarray]:
        """Get a MAP estimation from logp.
        logp: num_RNA_species by row by col OR num_RNA_species by num_pixels

        Return:
        - s: index of RNA species assign
        - p: the corresponding log prob
        """
        s = np.argmax(logp, axis=axis)
        p = np.max(logp, axis=axis)
        return s, p

    def fast_logpostr(
        self, y: np.ndarray, r: np.ndarray, rows: np.ndarray, cols=np.ndarray
    ):
        y_1 = np.empty(shape=(r.shape[1], rows.shape[0]), dtype=np.float)
        y_0 = np.empty_like(y_1, dtype=np.float)
        eps = 1e-15
        for t in range(y.shape[0]):
            print(f"pid laylerwise likelihood on {t}th-turn ... ")
            t1, t2 = self.layerwise_likelihood(t, y[t])
            t1[t1 == 0] = eps
            t2[t2 == 0] = eps
            y_1[t] = np.log(t1[rows, cols])
            y_0[t] = np.log(t2[rows, cols])
        print(f"pid join the likelihood ... ")
        ll = np.matmul(r, y_1) + np.matmul(1 - r, y_0)
        print(f"pid add the log prior ... ")
        logp = np.repeat(self.logw, ll.shape[1]).reshape(ll.shape) + ll
        return logp, y_1, y_0
