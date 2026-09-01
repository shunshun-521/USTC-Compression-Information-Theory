"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


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

    def _hard_decision(self, L, R):
        total = L + R
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    idx1 = slice(i, i + s)
                    idx2 = slice(i + s, i + 2 * s)
                    L[idx1, stage] = _f_min_sum(
                        R[idx1, stage] + L[idx2, stage + 1],
                        L[idx1, stage + 1],
                        self.alpha,
                    )
                    L[idx2, stage] = (
                        _f_min_sum(R[idx1, stage], L[idx1, stage + 1], self.alpha)
                        + L[idx2, stage + 1]
                    )

            for stage in range(n):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    idx1 = slice(i, i + s)
                    idx2 = slice(i + s, i + 2 * s)
                    R[idx1, stage + 1] = _f_min_sum(
                        R[idx2, stage] + L[idx2, stage + 1],
                        R[idx1, stage],
                        self.alpha,
                    )
                    R[idx2, stage + 1] = (
                        _f_min_sum(R[idx1, stage], L[idx1, stage + 1], self.alpha)
                        + R[idx2, stage]
                    )

            u_hat = self._hard_decision(L[:, 0], R[:, 0])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            x_hard = hard_ch[self.br]
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        u_hat = self._hard_decision(L[:, 0], R[:, 0])
        return u_hat.astype(int), num_iters
