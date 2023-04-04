import numpy as np


def solve(y, K):
    """
    Returns the least-squares solution.

    Returns the state `x` to the equation ``y = Kx`` given the measurements `y` and the 
    forward model matrix `K`.

    Parameters
    ----------
    y : (M,) array_like
        Measurement vector.
    K : (M, N,) array_like
        Forward matrix.

    Returns
    -------
    x : (N,) ndarray
        Approximate solution for the state.
    """
    x, res, rank, s = np.linalg.lstsq(K, y, rcond=None)
    return x
