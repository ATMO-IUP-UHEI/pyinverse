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
        forward_model: np.ndarray | sparse.COO,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> None:
        """
        Bayesian loss function containing the information needed for the analytical
        solution based on the implementation of Yadav and Michalak:
        https://doi.org/10.5194/gmd-6-583-2013

        t: temporal dimension of state space
        s: spatial dimension of state space
        n: number of measurements

        Args:
            prior (np.ndarray): Prior state (t, s)
            prior_standard_deviation (np.ndarray): Prior standard deviation (t, s)
            prior_temporal_correlation (np.ndarray): Temporal part of the correlation
             matrix (t, t)
            prior_spatial_correlation (np.ndarray): Spatial part of the correlation
             matrix (s, s)
            forward_model (np.ndarray | sparse.COO): Forward model (n, t, s)
            measurement (np.ndarray): Measurements (n)
            measurement_covariance (np.ndarray): Full covariance matrix of the
             measurements (n, n)
        """
        self.prior = prior
        self.prior_standard_deviation = prior_standard_deviation
        self.prior_temporal_correlation = prior_temporal_correlation
        self.prior_spatial_correlation = prior_spatial_correlation
        self.forward_model = forward_model
        self.measurement = measurement
        self.measurement_covariance = measurement_covariance
        self.check_shapes()

    def check_shapes(self):
        shape_problems = ""
        if len(self.prior.shape) != 2:
            shape_problems += "Prior should be a 2D array. "
        if len(self.prior_standard_deviation.shape) != 2:
            shape_problems += "Prior standard deviation should be a 2D array. "
        if len(self.prior_temporal_correlation.shape) != 2:
            shape_problems += "Prior temporal correlation should be a 2D array. "
        if len(self.prior_spatial_correlation.shape) != 2:
            shape_problems += "Prior spatial correlation should be a 2D array. "
        if len(self.forward_model.shape) != 3:
            shape_problems += "Forward model should be a 3D array. "
        if len(self.measurement.shape) != 1:
            shape_problems += "Measurement should be a 1D array. "
        if len(self.measurement_covariance.shape) != 2:
            shape_problems += "Measurement covariance should be a 2D array. "
        
        measurement_size = len(self.measurement)
        spatial_state_size = self.prior_spatial_correlation.shape[0]
        temporal_state_size = self.prior_temporal_correlation.shape[0]

        if self.prior.shape != (temporal_state_size, spatial_state_size):
            shape_problems += (
                "Prior shape is incorrect. Expected: "
                f"{(temporal_state_size, spatial_state_size)}, got: {self.prior.shape}. "
            )
        if self.prior_standard_deviation.shape != (
            temporal_state_size, spatial_state_size
        ):
            shape_problems += (
                "Prior standard deviation shape is incorrect. Expected: "
                f"{(temporal_state_size, spatial_state_size)}, got: "
                f"{self.prior_standard_deviation.shape}. "
            )
        if self.prior_temporal_correlation.shape != (
            temporal_state_size, temporal_state_size
        ):
            shape_problems += (
                "Prior temporal correlation shape is incorrect. Expected: "
                f"{(temporal_state_size, temporal_state_size)}, got: "
                f"{self.prior_temporal_correlation.shape}. "
            )
        if self.prior_spatial_correlation.shape != (
            spatial_state_size, spatial_state_size
        ):
            shape_problems += (
                "Prior spatial correlation shape is incorrect. Expected: "
                f"{(spatial_state_size, spatial_state_size)}, got: "
                f"{self.prior_spatial_correlation.shape}. "
            )
        if self.forward_model.shape != (
            measurement_size, temporal_state_size, spatial_state_size
        ):
            shape_problems += (
                "Forward model shape is incorrect. Expected: "
                f"{(measurement_size, temporal_state_size, spatial_state_size)}, got: "
                f"{self.forward_model.shape}. "
            )
        if shape_problems:
            raise ValueError(shape_problems)
        return
