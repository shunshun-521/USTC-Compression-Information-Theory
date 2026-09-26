class PolarCode:
    def __init__(self, N):
        import numpy as np

        self.N = N
        self.n = int(np.log2(N))
        self.u = np.zeros(N, dtype=int)
        self.likelihoods = np.zeros(N, dtype=np.float64)
        self.frozen = []
