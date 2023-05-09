from pyinverse.solver import LSTSQ
import numpy as np


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
