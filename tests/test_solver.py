from pyinverse.solver import LSTSQ, BayesianAnalytical
from pyinverse.loss import Bayesian
import numpy as np
import pytest


@pytest.fixture
def bayesianloss():
    rng = np.random.default_rng(seed=0)
    n = 10
    m = 5
    x_prior = rng.normal(size=n)
    cov_prior = rng.normal(size=(n, n))
    cov_prior = cov_prior @ cov_prior.T
    K = rng.normal(size=(m, n))
    y = K @ x_prior + rng.normal(size=m)
    cov_y = rng.normal(size=m)
    cov_y = cov_y @ cov_y.T
    loss = Bayesian(y=y, cov_y=cov_y, K=K, x_prior=x_prior, cov_prior=cov_prior)
    return loss


class DummyLoss:
    def __init__(self, K, y):
        self.K = K
        self.y = y

    def get_K(self):
        return self.K

    def get_y(self):
        return self.y


def test_lstsq():
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
        K = rng.normal(size=(m, n))
        y = K @ x

        loss = DummyLoss(K, y)
        solver = LSTSQ(loss)

        x_est, res, rank, s = solver()

        assert x_est.shape == x.shape


def test_bayesian_analytical(bayesianloss: Bayesian):
    solver = BayesianAnalytical(bayesianloss)
    x_posterior, cov_posterior = solver()
    averaging_kernel = solver.averaging_kernel
    gain = solver.gain
    std_posterior = solver.std_posterior
    y_posterior = solver.y_posterior
    assert x_posterior.shape == bayesianloss.x_prior.shape
    assert cov_posterior.shape == bayesianloss.cov_prior.shape
    assert averaging_kernel.shape == bayesianloss.cov_prior.shape
    assert gain.shape == bayesianloss.K.T.shape
    assert std_posterior.shape == bayesianloss.x_prior.shape
    assert y_posterior.shape == bayesianloss.y.shape
    dummyloss = DummyLoss(bayesianloss.K, bayesianloss.y)
    with pytest.raises(TypeError):
        solver = BayesianAnalytical(dummyloss)


    