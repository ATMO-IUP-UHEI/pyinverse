from pyinverse import utils
import numpy as np


def test_whitening():
    # Create forward model
    cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]])
    cov_strong_correlation = np.array(
        [[1.0, 0.9, 0.9], [0.9, 1.0, 1.0], [0.9, 1.0, 1.0]]
    )
    # Create random state
    rng = np.random.default_rng(seed=0)
    n = 3
    x = rng.normal(size=n)
    # For the strong correlation the values have to actually be correlated
    x_strong_correlation = x[[0,1, 1]]
    ranks = [3, 2]
    covs = [cov, cov_strong_correlation]
    xs = [x, x_strong_correlation]
    for cov, rank, x in zip(covs, ranks, xs):
        # Check that cov is hermitian
        assert np.allclose(cov.T, cov)

        # Create whitening
        whitening = utils.Whitening(cov)

        # Check that the rank of cov is correct
        assert whitening.rank == rank

        # As cov is symmetric, u and vh should be equal
        assert np.allclose(whitening.u.T, whitening.vh)

        # Check that the whitening of cov is the identity matrix
        assert np.allclose(whitening.transform(cov), np.eye(whitening.rank))

        # Check that the inverse whitening of the identity matrix is cov
        assert np.allclose(whitening.inverse_transform(np.eye(whitening.rank)), cov)

        # Transform state
        x_white = whitening.transform(x)

        # Check that the transformed state has the correct shape
        assert x_white.shape == (whitening.rank,)

        # Check that the inverse transform of the transformed state is the original state
        assert np.allclose(whitening.inverse_transform(x_white), x)
        