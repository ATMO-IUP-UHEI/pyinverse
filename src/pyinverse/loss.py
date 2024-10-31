import numpy as np
import sparse


class LeastSquares:
    def __init__(self, y, K):
        self.y = y
        self.K = K

    def __call__(self, x):
        diff = self.y - self.K @ x
        measurement_loss = diff @ diff
        return measurement_loss

    def get_K(self):
        return self.K

    def get_y(self):
        return self.y


class Thikonov:
    def __init__(self, y, K, alpha):
        self.y = y
        self.K = K
        self.alpha = alpha

    def __call__(self, x):
        diff = self.y - self.K @ x
        measurement_loss = diff @ diff
        regularization_loss = self.alpha * x @ x
        return measurement_loss + regularization_loss

    def get_K(self):
        m, n = self.K.shape
        reg = np.sqrt(self.alpha) * np.eye(n)
        K_reg = np.append(self.K, reg, axis=0)
        return K_reg

    def get_y(self):
        (m,) = self.y.shape
        reg = np.zeros(m)
        y_reg = np.append(self.y, reg, axis=0)
        return y_reg


class Bayesian:
    def __init__(self, y, cov_y, K, x_prior, cov_prior):
        self.y = y
        # Make sure cov_y is a matrix
        dim = len(cov_y.shape)
        if dim == 2:
            # Covariance is already a matrix
            pass
        elif dim == 1:
            # Covariance is a vector
            cov_y = np.diag(cov_y)
        self.cov_y = cov_y
        self.K = K
        self.x_prior = x_prior
        dim = len(cov_prior.shape)
        if dim == 2:
            # Prior is already a matrix
            pass
        elif dim == 1:
            # Prior is a vector
            cov_prior = np.diag(cov_prior)
        self.cov_prior = cov_prior

    def __call__(self, x):
        diff = self.y - self.K @ x
        measurement_loss = diff.T @ np.linalg.inv(self.cov_y) @ diff
        diff = x - self.x_prior
        prior_loss = diff.T @ np.linalg.inv(self.cov_prior) @ diff
        return measurement_loss + prior_loss

    def get_K(self):
        cov_y_sqrt_inv = np.linalg.inv(np.linalg.cholesky(self.cov_y))
        cov_prior_sqrt_inv = np.linalg.inv(np.linalg.cholesky(self.cov_prior))
        K_reg = np.append(cov_y_sqrt_inv @ self.K, cov_prior_sqrt_inv, axis=0)
        return K_reg

    def get_y(self):
        cov_y_sqrt_inv = np.linalg.inv(np.linalg.cholesky(self.cov_y))
        cov_prior_sqrt_inv = np.linalg.inv(np.linalg.cholesky(self.cov_prior))
        y_reg = np.append(
            cov_y_sqrt_inv @ self.y, cov_prior_sqrt_inv @ self.x_prior, axis=0
        )
        return y_reg

class BayesianYM:
    def __init__(
        self,
        prior: np.ndarray,
        prior_standard_deviation: np.ndarray,
        prior_temporal_correlation: np.ndarray,
        prior_spatial_correlation: np.ndarray,
        forward_model: np.ndarray|sparse.COO,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray
    ) -> None:
        self.prior = prior
        self.prior_standard_deviation = prior_standard_deviation
        self.prior_temporal_correlation = prior_temporal_correlation
        self.prior_spatial_correlation = prior_spatial_correlation
        self.forward_model = forward_model
        self.measurement = measurement
        self.measurement_covariance = measurement_covariance
        self.check_shapes()

    def check_shapes(self):
        assert len(self.prior.shape) == 2, "Prior should be a 2D array"
        assert len(self.prior_standard_deviation.shape) == 2, "Prior standard deviation should be a 2D array"
        assert len(self.prior_temporal_correlation.shape) == 2, "Prior temporal correlation should be a 2D array"
        assert len(self.prior_spatial_correlation.shape) == 2, "Prior spatial correlation should be a 2D array"
        assert len(self.forward_model.shape) == 3, "Forward model should be a 3D array"
        assert len(self.measurement.shape) == 1, "Measurement should be a 1D array"
        assert len(self.measurement_covariance.shape) == 2, "Measurement covariance should be a 2D array"

        measurement_size = len(self.measurement)
        spatial_state_size = self.prior_spatial_correlation.shape[0]
        temporal_state_size = self.prior_temporal_correlation.shape[0]

        assert self.prior.shape == (temporal_state_size, spatial_state_size), f"Prior shape is incorrect. Expected: {(temporal_state_size, spatial_state_size)}, got: {self.prior.shape}"
        assert self.prior_standard_deviation.shape == (temporal_state_size, spatial_state_size), f"Prior standard deviation shape is incorrect. Expected: {(temporal_state_size, spatial_state_size)}, got: {self.prior_standard_deviation.shape}"
        assert self.prior_temporal_correlation.shape == (temporal_state_size, temporal_state_size), f"Prior temporal correlation shape is incorrect. Expected: {(temporal_state_size, temporal_state_size)}, got: {self.prior_temporal_correlation.shape}"
        assert self.prior_spatial_correlation.shape == (spatial_state_size, spatial_state_size), f"Prior spatial correlation shape is incorrect. Expected: {(spatial_state_size, spatial_state_size)}, got: {self.prior_spatial_correlation.shape}"
        assert self.forward_model.shape == (measurement_size, temporal_state_size, spatial_state_size), f"Forward model shape is incorrect. Expected: {(measurement_size, temporal_state_size, spatial_state_size)}, got: {self.forward_model.shape}"
        assert self.measurement_covariance.shape == (measurement_size, measurement_size), f"Measurement covariance shape is incorrect. Expected: {(measurement_size, measurement_size)}, got: {self.measurement_covariance.shape}"
        return
