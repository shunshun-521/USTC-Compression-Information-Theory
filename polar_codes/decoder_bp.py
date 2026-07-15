"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation, _permute_channel_llrs
from encoder import polar_encode
from channel import hard_decision_llr

_LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

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
        llr = _permute_channel_llrs(llr_ch, self.N)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        L[idx_i, j - 1] = self._f_min_sum(
                            R[idx_i, j] + L[idx_is, j + 1], L[idx_i, j + 1]
                        )
                        L[idx_is, j - 1] = self._f_min_sum(
                            R[idx_i, j], L[idx_i, j + 1]
                        ) + L[idx_is, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        R[idx_i, j + 1] = self._f_min_sum(
                            R[idx_is, j] + L[idx_is, j + 1], R[idx_i, j]
                        )
                        R[idx_is, j + 1] = self._f_min_sum(
                            R[idx_i, j], L[idx_i, j + 1]
                        ) + R[idx_is, j]

            num_iters = it

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
