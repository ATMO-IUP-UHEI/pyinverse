import numpy as np
import sparse

from pyinverse.loss import Bayesian, BayesianYM
from typing import Tuple, List
from numba import njit, prange



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

 


class BayesianAnalyticalYM_Base:
    def __init__(
        self,
        loss: BayesianYM
    ) -> None:
        self.loss = loss
        self._hqh = None
        self._hq = None
        self._hx_prior = None
        self._gain = None
        self._x_posterior = None

    @property
    def gain(self) -> np.ndarray:
        if self._gain is None:
            self._gain = (
                np.tensordot(
                    self.hq,
                    np.linalg.inv(self.hqh + self.loss.measurement_covariance),
                    axes=([0], [0])
                )
            )
        return self._gain
    
    @property
    def x_posterior(self) -> np.ndarray:
        if self._x_posterior is None:
            self._x_posterior = self.loss.prior + np.tensordot(
                self.gain, 
                self.loss.measurement - self.hx_prior,
                axes=([2], [0])
            )
        return self._x_posterior

    @property
    def hqh(self):
        if self._hqh is None:
            self._hq, self._hqh = self.compute_hq_and_hqh(
                self.loss.forward_model,
                self.loss.prior_temporal_correlation,
                self.loss.prior_spatial_correlation,
                self.loss.prior_standard_deviation,
            )
        return self._hqh

    @property
    def hq(self):
        if self._hq is None:
            self._hq, self._hqh = self.compute_hq_and_hqh(
                self.loss.forward_model,
                self.loss.prior_temporal_correlation,
                self.loss.prior_spatial_correlation,
                self.loss.prior_standard_deviation,
            )
        return self._hq

    @staticmethod
    @njit
    def compute_hq_and_hqh(
        forward_model: np.ndarray, 
        temporal_correlation: np.ndarray, 
        spatial_correlation: np.ndarray, 
        standard_deviation: np.ndarray,
    ) -> np.ndarray:

        measurement_shape, time_shape, space_shape = forward_model.shape

        hq_result = np.zeros((measurement_shape, time_shape, space_shape), dtype=np.float32)
        hqh_result = np.zeros((measurement_shape, measurement_shape), dtype=np.float32)
        for j in range(time_shape):
            partial_result  = np.zeros((measurement_shape, space_shape), dtype=np.float32)
            for i in range(time_shape):
                temporal_correlation_ij = temporal_correlation[i, j]
                if temporal_correlation_ij == 0:
                    continue
                partial_result += forward_model[:, i, :] * temporal_correlation_ij * standard_deviation[i]

            partial_result = np.dot(
                partial_result,
                spatial_correlation * standard_deviation[j]
            )

            hq_result[:, j, :] = partial_result

            hqh_result += np.dot(
                partial_result,
                forward_model[:, j, :].T
            )

        return hq_result, hqh_result

    @property
    def hx_prior(self):
        if self._hx_prior is None:
            self._hx_prior = np.tensordot(self.loss.forward_model, self.loss.prior, axes=([1, 2], [0, 1]))
        return self._hx_prior
    
    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_posterior

class BayesianAnalyticalYM_Sparse(BayesianAnalyticalYM_Base):
    def __init__(
        self,
        loss: BayesianYM
        ) -> None:
        super().__init__(loss)


    @property
    def hqh(self):
        if self._hqh is None:
            self._hq, self._hqh = self.compute_hq_and_hqh(
                *self.extract_sparse_data(self.loss.forward_model),
                self.loss.prior_temporal_correlation,
                self.loss.prior_spatial_correlation,
                self.loss.prior_standard_deviation,
            )
        return self._hqh

    @property
    def hq(self):
        if self._hq is None:
            self._hq, self._hqh = self.compute_hq_and_hqh(
                *self.extract_sparse_data(self.loss.forward_model),
                self.loss.prior_temporal_correlation,
                self.loss.prior_spatial_correlation,
                self.loss.prior_standard_deviation,
            )
        return self._hq
    
    @staticmethod
    @njit
    def compute_hq_and_hqh(
        forward_model_data: np.ndarray,
        measurement_coordinates: np.ndarray,
        temporal_coordinates: np.ndarray,
        spatial_coordinates: np.ndarray, 
        forward_model_shape: np.ndarray,
        temporal_correlation: np.ndarray, 
        spatial_correlation: np.ndarray, 
        standard_deviation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        measurement_shape, time_shape, space_shape = forward_model_shape
        hq_result = np.zeros((measurement_shape, time_shape, space_shape), dtype=np.float32)
        hqh_result = np.zeros((measurement_shape, measurement_shape), dtype=np.float32)
        
        unique_temporal_coordinates = np.unique(temporal_coordinates)
        for j in unique_temporal_coordinates:
            partial_result  = np.zeros((measurement_shape, space_shape), dtype=np.float32)
            for i in unique_temporal_coordinates:
                
                temporal_correlation_ij = temporal_correlation[i, j]
                if temporal_correlation_ij == 0:
                    continue

                forward_model_data_i = forward_model_data[temporal_coordinates == i]
                forward_model_rows_i = measurement_coordinates[temporal_coordinates == i]
                forward_model_columns_i = spatial_coordinates[temporal_coordinates == i]

                for k in prange(len(forward_model_data_i)):
                    forward_model_value = forward_model_data_i[k]
                    forward_model_row = forward_model_rows_i[k]
                    forward_model_column = forward_model_columns_i[k]
                    partial_result[forward_model_row, forward_model_column] += forward_model_value * temporal_correlation_ij * standard_deviation[i, forward_model_column]
            partial_result = np.dot(
                partial_result,
                spatial_correlation * standard_deviation[j]
            )

            hq_result[:, j, :] = partial_result

            forward_model_data_j = forward_model_data[temporal_coordinates == j]
            forward_model_rows_j = measurement_coordinates[temporal_coordinates == j]
            forward_model_columns_j = spatial_coordinates[temporal_coordinates == j]
            for k in range(len(forward_model_data_j)):
                forward_model_value = forward_model_data_j[k]
                forward_model_row = forward_model_rows_j[k]
                forward_model_column = forward_model_columns_j[k]
                for l in prange(measurement_shape):
                    hqh_result[l, forward_model_row] += partial_result[l, forward_model_column] * forward_model_value

        return hq_result, hqh_result

    @staticmethod
    def extract_sparse_data(forward_model: sparse.COO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            forward_model.data,
            forward_model.coords[0],
            forward_model.coords[1],
            forward_model.coords[2],
            forward_model.shape
        )
        
class BayesianAnalyticalYM:
    def __init__(
        self,
        loss: BayesianYM,
    ) -> None:
        self.loss = loss
        if isinstance(loss.forward_model, np.ndarray):
            self.solver = BayesianAnalyticalYM_Base(loss)
        elif isinstance(loss.forward_model, sparse.COO):
            self.solver = BayesianAnalyticalYM_Sparse(loss)
        else:
            raise TypeError(f"Unsupported forward model type in loss: {type(loss.forward_model)}")