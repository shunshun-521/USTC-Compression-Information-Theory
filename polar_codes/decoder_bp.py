"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = np.empty(self.N, dtype=np.float64)
        llr[self.br] = llr_ch

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for t in range(step):
                        idx_u = i + t
                        idx_l = i + t + step
                        L[idx_u, j - 1] = self._f_min_sum(
                            R[idx_u, j] + L[idx_l, j],
                            L[idx_u, j],
                        )
                        L[idx_l, j - 1] = self._f_min_sum(
                            R[idx_u, j],
                            L[idx_u, j],
                        ) + L[idx_l, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for t in range(step):
                        idx_u = i + t
                        idx_l = i + t + step
                        R[idx_u, j + 1] = self._f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1],
                            R[idx_u, j],
                        )
                        R[idx_l, j + 1] = self._f_min_sum(
                            R[idx_u, j],
                            L[idx_u, j + 1],
                        ) + R[idx_l, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
