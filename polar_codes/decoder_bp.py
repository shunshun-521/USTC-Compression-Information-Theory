"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _g_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        br = bit_reversal_permutation(N)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j] = _g_min_sum(
                            L[idx, j + 1],
                            L[idx + s, j + 1] + R[idx + s, j],
                            self.alpha,
                        )
                        L[idx + s, j] = (
                            _g_min_sum(R[idx, j], L[idx, j + 1], self.alpha)
                            + L[idx + s, j + 1]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = _g_min_sum(
                            R[idx, j],
                            L[idx + s, j + 1] + R[idx + s, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = (
                            _g_min_sum(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx + s, j]
                        )

            u_hat = self._hard_decision(L)
            if self._early_stop(u_hat, llr_ch, br):
                break

        u_hat = self._hard_decision(L)
        return u_hat, num_iters

    def _hard_decision(self, L):
        u_hat = (L[:, 0] < 0).astype(np.int32)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch, br):
        x_hat = polar_encode(u_hat)
        hard_nat = (llr_ch < 0).astype(np.int32)[br]
        return np.array_equal(x_hat, hard_nat)
