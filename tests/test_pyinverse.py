from pyinverse import pyinverse
import numpy as np

def test_solve():
    # Create forward model
    x = np.array([1,2])
    K = np.array([[1, 0], [0, 2]])
    y = K @ x
    x_solver = pyinverse.solve(y, K)
    assert np.allclose(x, x_solver)
