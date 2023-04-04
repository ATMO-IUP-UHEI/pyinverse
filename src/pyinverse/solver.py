import numpy as np


class LSTSQ:
    def __init__(self, loss, rcond=None):
        self.loss = loss
        self.rcond = rcond

    def __call__(self):
        self.K = self.loss.get_K()
        self.y = self.loss.get_y()
        return self.solve(K=self.K, y=self.y, rcond=self.rcond)

    @staticmethod
    def solve(K, y, rcond=None):
        x, res, rank, s = np.linalg.lstsq(a=K, b=y, rcond=rcond)
        return x, res, rank, s
