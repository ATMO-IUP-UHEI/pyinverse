from pyinverse.solver import (
    LSTSQ,
    BayesianAnalytical,
    BayesianAnalyticalYM_Base,
    BayesianAnalyticalYM_Sparse,
)
from pyinverse.loss import Bayesian, BayesianYM
import numpy as np
import sparse
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


@pytest.fixture
def bayesianloss_ym():
    rng = np.random.default_rng(seed=0)
    n = 10
    m_s = 7
    m_t = 8
    prior = rng.normal(size=(m_t, m_s)).astype(np.float32)
    prior_standard_deviation = rng.normal(size=(m_t, m_s)).astype(np.float32)
    prior_temporal_correlation = rng.normal(size=(m_t, m_t)).astype(np.float32)
    prior_spatial_correlation = rng.normal(size=(m_s, m_s)).astype(np.float32)
    forward_model = rng.normal(size=(n, m_t, m_s)).astype(np.float32)
    measurement = rng.normal(size=n).astype(np.float32)
    measurement_covariance = rng.normal(size=(n, n)).astype(np.float32)
    loss = BayesianYM(
        prior=prior,
        prior_standard_deviation=prior_standard_deviation,
        prior_temporal_correlation=prior_temporal_correlation,
        prior_spatial_correlation=prior_spatial_correlation,
        forward_model=forward_model,
        measurement=measurement,
        measurement_covariance=measurement_covariance,
    )
    return loss


@pytest.fixture
def compute_hq_and_hqh_args(bayesianloss_ym: BayesianYM):
    footprint_values = bayesianloss_ym.forward_model.flatten()
    dummy = np.ones(bayesianloss_ym.forward_model.shape, dtype=int)
    measurement_coordinates = (
        np.arange(10, dtype=int)[:, None, None] * dummy
    ).flatten()
    time_coordinates = (np.arange(8, dtype=int)[None, :, None] * dummy).flatten()
    space_coordinates = (np.arange(7, dtype=int)[None, None, :] * dummy).flatten()
    footprint_shape = np.array(bayesianloss_ym.forward_model.shape)
    return (
        footprint_values,
        measurement_coordinates,
        time_coordinates,
        space_coordinates,
        footprint_shape,
    )


@pytest.fixture
def bayesianloss_ym_sparse():
    rng = np.random.default_rng(seed=0)
    n = 10
    m_s = 7
    m_t = 8
    prior = rng.normal(size=(m_t, m_s)).astype(np.float32)
    prior_standard_deviation = rng.normal(size=(m_t, m_s)).astype(np.float32)
    prior_temporal_correlation = rng.normal(size=(m_t, m_t)).astype(np.float32)
    prior_spatial_correlation = rng.normal(size=(m_s, m_s)).astype(np.float32)
    forward_model = sparse.random(
        (n, m_t, m_s), density=0.1, format="coo", random_state=0
    ).astype(np.float32)
    measurement = rng.normal(size=n).astype(np.float32)
    measurement_covariance = rng.normal(size=(n, n)).astype(np.float32)
    loss = BayesianYM(
        prior=prior,
        prior_standard_deviation=prior_standard_deviation,
        prior_temporal_correlation=prior_temporal_correlation,
        prior_spatial_correlation=prior_spatial_correlation,
        forward_model=forward_model,
        measurement=measurement,
        measurement_covariance=measurement_covariance,
    )
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


class Test_BayesianAnalyticalYM_Base:
    def test_hq_hqh(self, bayesianloss_ym: BayesianYM):
        solver = BayesianAnalyticalYM_Base(bayesianloss_ym)
        hq = solver.hq
        hqh = solver.hqh
        footprint = bayesianloss_ym.forward_model
        prior_covariance = (
            bayesianloss_ym.prior_temporal_correlation[:, None, :, None]
            * bayesianloss_ym.prior_spatial_correlation[None, :, None, :]
            * bayesianloss_ym.prior_standard_deviation[..., None, None]
            * bayesianloss_ym.prior_standard_deviation[None, None, ...]
        )
        hq_expected = np.tensordot(footprint, prior_covariance, axes=([1, 2], [0, 1]))
        hqh_expected = np.tensordot(
            hq_expected, footprint.transpose((1, 2, 0)), axes=([1, 2], [0, 1])
        )
        assert np.allclose(hq, hq_expected, atol=0, rtol=1e-4)
        assert np.allclose(hqh, hqh_expected, atol=0, rtol=1e-4)


class Test_BayesianAnalyticalYM_Sparse:
    def test_compute_hq_and_hqh(
        self, compute_hq_and_hqh_args, bayesianloss_ym: BayesianYM
    ):
        hq, hqh = BayesianAnalyticalYM_Sparse.compute_hq_and_hqh(
            *compute_hq_and_hqh_args,
            bayesianloss_ym.prior_temporal_correlation,
            bayesianloss_ym.prior_spatial_correlation,
            bayesianloss_ym.prior_standard_deviation
        )

        footprint = bayesianloss_ym.forward_model
        prior_covariance = (
            bayesianloss_ym.prior_temporal_correlation[:, None, :, None]
            * bayesianloss_ym.prior_spatial_correlation[None, :, None, :]
            * bayesianloss_ym.prior_standard_deviation[..., None, None]
            * bayesianloss_ym.prior_standard_deviation[None, None, ...]
        )
        hq_expected = np.tensordot(footprint, prior_covariance, axes=([1, 2], [0, 1]))
        hqh_expected = np.tensordot(
            hq_expected, footprint.transpose((1, 2, 0)), axes=([1, 2], [0, 1])
        )
        assert np.allclose(hq, hq_expected, atol=0, rtol=1e-4)
        assert np.allclose(hqh, hqh_expected, atol=0, rtol=1e-4)

    def test_hq_hqh(self, bayesianloss_ym_sparse: BayesianYM):
        solver = BayesianAnalyticalYM_Sparse(bayesianloss_ym_sparse)
        hq = solver.hq
        hqh = solver.hqh
        footprint = bayesianloss_ym_sparse.forward_model.todense()
        prior_covariance = (
            bayesianloss_ym_sparse.prior_temporal_correlation[:, None, :, None]
            * bayesianloss_ym_sparse.prior_spatial_correlation[None, :, None, :]
            * bayesianloss_ym_sparse.prior_standard_deviation[..., None, None]
            * bayesianloss_ym_sparse.prior_standard_deviation[None, None, ...]
        )
        hq_expected = np.tensordot(footprint, prior_covariance, axes=([1, 2], [0, 1]))
        hqh_expected = np.tensordot(
            hq_expected, footprint.transpose((1, 2, 0)), axes=([1, 2], [0, 1])
        )
        assert np.allclose(hq, hq_expected, atol=1e-7, rtol=1e-4)
        assert np.allclose(hqh, hqh_expected, atol=1e-7, rtol=1e-4)
