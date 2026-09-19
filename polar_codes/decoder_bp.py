"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = (llr_ch < 0).astype(np.int8)
        return np.array_equal(x_hat, x_hard)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_br
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j - 1] = _f_min_sum(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _f_min_sum(R[i0, j], L[i0, j], self.alpha) + L[
                            i1, j
                        ]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i0, j + 1] = _f_min_sum(
                            R[i1, j] + L[i1, j + 1], R[i0, j], self.alpha
                        )
                        R[i1, j + 1] = (
                            _f_min_sum(R[i0, j], L[i0, j + 1], self.alpha) + R[i1, j]
                        )

            u_hat = self._hard_decision(L, R)
            num_iters = it
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
