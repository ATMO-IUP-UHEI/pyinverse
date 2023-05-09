from pyinverse.forward_model import Linear
import numpy as np

def test_linear():
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

        forward_model = Linear(K)
        y_test = forward_model(x)
        assert np.allclose(y, y_test)
