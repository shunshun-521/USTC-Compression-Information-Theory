"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（极化因子图，min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        br = bit_reversal_permutation(N)
        llr_proc = llr_nat[br]
        hard_nat = (llr_nat < 0).astype(int)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_proc
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1
            for j in range(n - 1, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j + 1], L[idx, j + 1]
                        )
                        L[idx + s, j - 1] = self._f_min_sum(
                            R[idx, j], L[idx, j + 1]
                        ) + L[idx + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_min_sum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx + s, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_nat):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
