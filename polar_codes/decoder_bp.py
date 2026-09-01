"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i : i + s, j]
                    Lis = L[i + s : i + 2 * s, j]
                    Ri = R[i : i + s, j]
                    Ris = R[i + s : i + 2 * s, j]
                    L[i : i + s, j - 1] = self._f_min_sum(Ri + Lis, Li)
                    L[i + s : i + 2 * s, j - 1] = self._f_min_sum(Ri, Li) + Ris

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li = L[i : i + s, j + 1]
                    Lis = L[i + s : i + 2 * s, j + 1]
                    Ri = R[i : i + s, j]
                    Ris = R[i + s : i + 2 * s, j]
                    R[i : i + s, j + 1] = self._f_min_sum(Ris + Lis, Ri)
                    R[i + s : i + 2 * s, j + 1] = self._f_min_sum(Ri, Li) + Ris

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat, self.max_iter
