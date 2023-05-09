from pyinverse.loss import LeastSquares, Thikonov, Bayesian
import numpy as np


def test():
    rng = np.random.default_rng(seed=0)
    n_tests = 100
    for i in range(n_tests):
        # State dimension
        n_min, n_max = (1, 100)
        n = rng.integers(n_min, n_max)
        # Measurement dimension
        m_min, m_max = (1, 100)
        m = rng.integers(m_min, m_max)

        x = rng.normal(size=n)
        x_pred = rng.normal(size=n)
        K = rng.normal(size=(m, n))
        y = K @ x

        alpha = rng.normal()

        measurement_loss = np.linalg.norm(y - K @ x_pred) ** 2
        regularization_loss = alpha * np.linalg.norm(x_pred) ** 2

        least_squares_test(x_pred, K, y, measurement_loss)
        thikonov_test(x_pred, K, y, alpha, measurement_loss, regularization_loss)


def least_squares_test(x_pred, K, y, measurement_loss):
    loss = LeastSquares(y, K)
    assert np.allclose(measurement_loss, loss(x_pred))
    assert np.allclose(y, loss.get_y())
    assert np.allclose(K, loss.get_K())


def thikonov_test(x_pred, K, y, alpha, measurement_loss, regularization_loss):
    loss = Thikonov(y, K, alpha)
    assert np.allclose(measurement_loss + regularization_loss, loss(x_pred))
    # assert np.allclose(y, loss.get_y())
    # assert np.allclose(K, loss.get_K())
