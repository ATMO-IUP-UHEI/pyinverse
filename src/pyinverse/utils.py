import numpy as np


class Whitening:
    def __init__(self, cov, tol=None):
        self.cov = cov
        # Check that cov is hermitian
        assert np.allclose(cov.T, cov)
        u, s, vh = np.linalg.svd(cov, hermitian=True)
        self.u = u
        self.s = s
        self.vh = vh
        self.s_sqrt = np.sqrt(s)
        self.s_inv_sqrt = 1 / self.s_sqrt
        self.s_inv = 1 / s
        self.n = cov.shape[0]
        self.rank = np.linalg.matrix_rank(cov, tol=tol, hermitian=True)

    def transform(self, x):
        if len(x.shape) == 1:
            return ((self.u.T @ x) * self.s_inv_sqrt)[: self.rank]
        else:
            return (self.u.T @ x @ self.u * np.diag(self.s_inv))[
                : self.rank, : self.rank
            ]

    def inverse_transform(self, x):
        if len(x.shape) == 1:
            return self.u @ (np.pad(x, (0, self.n - self.rank)) * self.s_sqrt)
        else:
            return (
                self.u
                @ (
                    np.pad(x, ((0, self.n - self.rank), (0, self.n - self.rank)))
                    * np.diag(self.s)
                )
                @ self.u.T
            )
