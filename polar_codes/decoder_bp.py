"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode

LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        idx = i + k
                        idx2 = idx + stride
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx2, j], L[idx, j]
                        )
                        L[idx2, j - 1] = self._f_min_sum(R[idx, j], L[idx, j]) + L[idx2, j]

            for j in range(0, n):
                stride = 1 << j
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        idx = i + k
                        idx2 = idx + stride
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j]
                        )
                        R[idx2, j + 1] = self._f_min_sum(R[idx, j], L[idx, j + 1]) + R[idx2, j]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
