from pyinverse.loss import LeastSquares, Tikhonov, Bayesian, BayesianYM
import numpy as np
import pytest


def test_leastsquares_tikhonov_bayesian():
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
        x_prior = rng.normal(size=n)
        cov_prior = np.abs(rng.normal(size=n))
        K = rng.normal(size=(m, n))
        y = K @ x
        cov_y = np.abs(rng.normal(size=m))

        alpha = rng.normal()

        measurement_loss = np.linalg.norm(y - K @ x_pred) ** 2
        regularization_loss = alpha * np.linalg.norm(x_pred) ** 2

        least_squares_test(x_pred, K, y, measurement_loss)
        tikhonov_test(x_pred, K, y, alpha, measurement_loss, regularization_loss)

        measurement_loss = np.linalg.norm((y - K @ x_pred) / np.sqrt(cov_y)) ** 2
        regularization_loss = (
            np.linalg.norm((x_pred - x_prior) / np.sqrt(cov_prior)) ** 2
        )

        bayesian_test(
            x_pred,
            K,
            y,
            cov_y,
            x_prior,
            cov_prior,
            measurement_loss,
            regularization_loss,
        )


def least_squares_test(x_pred, K, y, measurement_loss):
    loss = LeastSquares(y, K)
    assert np.allclose(measurement_loss, loss(x_pred))
    assert np.allclose(y, loss.get_y())
    assert np.allclose(K, loss.get_K())


def tikhonov_test(x_pred, K, y, alpha, measurement_loss, regularization_loss):
    loss = Tikhonov(y, K, alpha)
    assert np.allclose(measurement_loss + regularization_loss, loss(x_pred))

    loss.get_y()
    loss.get_K()

    # assert np.allclose(y, loss.get_y())
    # assert np.allclose(K, loss.get_K())


def bayesian_test(
    x_pred, K, y, cov_y, x_prior, cov_prior, measurement_loss, regularization_loss
):
    loss = Bayesian(y, cov_y, K, x_prior, cov_prior)
    print(measurement_loss + regularization_loss)
    assert np.allclose(measurement_loss + regularization_loss, loss(x_pred))

    loss.get_y()
    loss.get_K()


class TestBayesianYM:
    def test_bayesian_ym_valid(self):
        n_measurements = 10
        n_t = 5
        n_s = 3
        prior = np.zeros((n_t, n_s))
        prior_standard_deviation = np.ones((n_t, n_s))
        prior_temporal_correlation = np.eye(n_t)
        prior_spatial_correlation = np.eye(n_s)
        forward_model = np.ones((n_measurements, n_t, n_s))
        measurement = np.ones(n_measurements)
        measurement_covariance = np.eye(n_measurements)
        loss = BayesianYM(
            prior,
            prior_standard_deviation,
            prior_temporal_correlation,
            prior_spatial_correlation,
            forward_model,
            measurement,
            measurement_covariance,
        )
        assert loss.prior is not None
        assert loss.prior_standard_deviation is not None
        assert loss.prior_temporal_correlation is not None
        assert loss.prior_spatial_correlation is not None
        assert loss.forward_model is not None
        assert loss.measurement is not None
        assert loss.measurement_covariance is not None

    def test_bayesian_ym_invalid(self):
        n_measurements = 10
        n_t = 5
        n_s = 3
        prior = np.zeros((n_s, n_t)) # Switched dimensions
        prior_standard_deviation = np.ones((n_t)) # missing dimension
        prior_temporal_correlation = np.eye(n_t)
        prior_spatial_correlation = np.eye(n_s)
        forward_model = np.ones((n_t, n_s, n_measurements)) # Switched dimensions
        measurement = np.ones(n_measurements)
        measurement_covariance = np.ones(n_measurements) # missing dimension
        try:
            loss = BayesianYM(
                prior,
                prior_standard_deviation,
                prior_temporal_correlation,
                prior_spatial_correlation,
                forward_model,
                measurement,
                measurement_covariance,
            )
        except ValueError as e:
            error_message = str(e)
        assert error_message.count(".") == 8
        assert "Prior shape is incorrect." in error_message
        assert "Prior standard deviation should be a 2D array. " in error_message
        assert (
            "Prior standard deviation shape is incorrect. Expected: (5, 3)" in 
            error_message
        )
        assert "Forward model shape is incorrect. Expected: (10, 5, 3)" in error_message
        assert "Measurement covariance should be a 2D array. " in error_message

    def test_bayesian_ym_invalid_2(self):
        n_measurements = 10
        n_t = 5
        n_s = 3
        prior = np.zeros((n_measurements)) # wrong dimension/missing dimension
        prior_standard_deviation = np.ones((n_t, n_s)) 
        prior_temporal_correlation = np.ones(n_t) # missing dimension
        prior_spatial_correlation = np.ones(n_s) # missing dimension
        forward_model = np.ones((n_t, n_measurements)) # missing dimension
        measurement = np.eye(n_measurements) # additional dimension
        measurement_covariance = np.eye(n_measurements) 

        try:
            loss = BayesianYM(
                prior,
                prior_standard_deviation,
                prior_temporal_correlation,
                prior_spatial_correlation,
                forward_model,
                measurement,
                measurement_covariance,
            )
        except ValueError as e:
            error_message = str(e)
        assert "Prior should be a 2D array. " in error_message
        assert "Prior temporal correlation should be a 2D array. " in error_message
        assert "Prior spatial correlation should be a 2D array. " in error_message
        assert "Forward model should be a 3D array. " in error_message
        assert "Measurement should be a 1D array. " in error_message
        assert "Prior shape is incorrect. Expected: (5, 3)" in error_message
        assert "Prior temporal correlation shape is incorrect. Expected: (5, 5)" in error_message
        assert "Prior spatial correlation shape is incorrect. Expected: (3, 3)" in error_message
        assert "Forward model shape is incorrect. Expected: (10, 5, 3)" in error_message
        
