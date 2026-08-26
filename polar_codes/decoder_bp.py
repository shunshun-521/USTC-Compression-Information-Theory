"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation

_LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = _LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[j - 1, idx] = self._f_min_sum(
                            R[j, idx] + L[j, idx + s], L[j, idx]
                        )
                        L[j - 1, idx + s] = self._f_min_sum(
                            R[j, idx], L[j, idx]
                        ) + L[j, idx + s]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[j + 1, idx] = self._f_min_sum(
                            R[j + 1, idx + s] + L[j + 1, idx + s], R[j, idx]
                        )
                        R[j + 1, idx + s] = self._f_min_sum(
                            R[j, idx], L[j + 1, idx]
                        ) + R[j + 1, idx + s]

            total_llr = L[0, :] + R[0, :]
            u_hat = np.where(total_llr >= 0, 0, 1)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[0, :] + R[0, :]
        u_hat = np.where(total_llr >= 0, 0, 1)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
