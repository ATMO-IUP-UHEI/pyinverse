import numpy as np
from pyinverse.loss import Bayesian
from typing import Tuple


class LSTSQ:
    def __init__(self, loss, rcond=None):
        self.loss = loss
        self.rcond = rcond

    def __call__(self):
        self.K = self.loss.get_K()
        self.y = self.loss.get_y()
        return self.solve(K=self.K, y=self.y, rcond=self.rcond)

    @staticmethod
    def solve(K, y, rcond=None):
        x, res, rank, s = np.linalg.lstsq(a=K, b=y, rcond=rcond)
        return x, res, rank, s


class BayesianAnalytical:
    def __init__(self, loss: Bayesian) -> None:
        """
        Analytical Bayesian inversion

        Solves for posterior and its covariance based on the given loss. Additionally
        grants access to the gain matrix and averaging kernel.

        Args:
            loss (Bayesian): Bayesian loss function
        """
        if not isinstance(loss, Bayesian):
            raise TypeError("Loss function must be Bayesian")
        self.loss = loss
        self._gain = None
        self._averaging_kernel = None
        self._x_posterior = None
        self._cov_posterior = None

    @property
    def gain(self) -> np.ndarray:
        if self._gain is None:
            self._gain = (
                self.loss.cov_prior
                @ self.loss.K.T
                @ np.linalg.inv(
                    self.loss.K @ self.loss.cov_prior @ self.loss.K.T + self.loss.cov_y
                )
            )
        return self._gain

    @property
    def averaging_kernel(self) -> np.ndarray:
        if self._averaging_kernel is None:
            self._averaging_kernel = self.gain @ self.loss.K
        return self._averaging_kernel

    @property
    def x_posterior(self) -> np.ndarray:
        if self._x_posterior is None:
            self._x_posterior = self.loss.x_prior + self.gain @ (
                self.loss.y - self.loss.K @ self.loss.x_prior
            )
        return self._x_posterior

    @property
    def cov_posterior(self) -> np.ndarray:
        if self._cov_posterior is None:
            self._cov_posterior = (
                self.loss.cov_prior - self.averaging_kernel @ self.loss.cov_prior
            )
        return self._cov_posterior

    @property
    def std_posterior(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_posterior))

    @property
    def y_posterior(self) -> np.ndarray:
        return self.loss.K @ self.x_posterior

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_posterior, self.cov_posterior
