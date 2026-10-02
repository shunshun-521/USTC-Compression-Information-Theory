"""精简版 PolarCode（无 GUI 依赖）"""
import numpy as np
from vendor.utils import arikan_gen, inverse_set


class PolarCode:
    def __init__(self, M, K):
        self.N = M
        self.M = M
        self.K = K
        self.n = int(np.log2(M))
        self.F = arikan_gen(self.n)
        self.frozen = np.array([], dtype=int)
        self.frozen_lookup = np.zeros(M, dtype=int)
        self.u = np.zeros(M, dtype=int)
        self.x = np.zeros(M, dtype=int)
        self.construction_type = "ga"
        self.likelihoods = np.zeros(M, dtype=np.float64)
        self.T = None
        self.punct_flag = False

    def set_message(self, message):
        self.u = np.zeros(self.N, dtype=int)
        info = inverse_set(self.frozen, self.N)
        self.u[info] = message

    def get_codeword(self):
        return self.u.copy()

    def get_normalised_SNR(self, design_SNR):
        Eb_No = 10 ** (design_SNR / 10)
        return Eb_No * (self.K / self.M)

    def get_lut(self, my_set):
        lut = np.zeros(self.N, dtype=int)
        for idx in my_set:
            lut[int(idx)] = 1
        return lut
