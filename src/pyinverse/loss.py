import numpy as np


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
        cov_y_sqrt_inv = np.linalg.inv(np.cholesky(self.cov_y))
        cov_prior_sqrt_inv = np.linalg.inv(np.cholesky(self.cov_prior))
        K_reg = np.append(cov_y_sqrt_inv @ self.K, cov_prior_sqrt_inv, axis=0)
        return K_reg

    def get_y(self):
        cov_y_sqrt_inv = np.linalg.inv(np.cholesky(self.cov_y))
        cov_prior_sqrt_inv = np.linalg.inv(np.cholesky(self.cov_prior))
        y_reg = np.append(
            cov_y_sqrt_inv @ self.y, cov_prior_sqrt_inv @ self.x_prior, axis=0
        )
        return y_reg
