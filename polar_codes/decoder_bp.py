"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化因子图，列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)
        hard_ch = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = i + k
                        i2 = i + k + s
                        L[i1, j - 1] = _f_min_sum(
                            R[i1, j - 1] + L[i2, j],
                            L[i1, j],
                            self.alpha,
                        )
                        L[i2, j - 1] = (
                            _f_min_sum(R[i1, j - 1], L[i1, j], self.alpha)
                            + L[i2, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = i + k
                        i2 = i + k + s
                        R[i1, j + 1] = _f_min_sum(
                            R[i2, j] + L[i2, j + 1],
                            R[i1, j],
                            self.alpha,
                        )
                        R[i2, j + 1] = (
                            _f_min_sum(R[i1, j], L[i1, j + 1], self.alpha)
                            + R[i2, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat[self._br], hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
