class Linear:
    def __init__(self, K):
        self.K = K

    def __call__(self, x):
        y = self.K @ x
        return y
