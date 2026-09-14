"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _hard_decision_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                span = step << 1
                for block in range(0, N, span):
                    for i in range(step):
                        idx = block + i
                        idx2 = idx + step
                        L[idx, j - 1] = _boxplus_minsum(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _boxplus_minsum(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                span = step << 1
                for block in range(0, N, span):
                    for i in range(step):
                        idx = block + i
                        idx2 = idx + step
                        R[idx, j + 1] = _boxplus_minsum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _boxplus_minsum(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx2, j]

            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = self._hard_decision_llr(llr_ch)
            num_iters = it
            if np.array_equal(x_hat, x_hard):
                break

        posterior = L[:, 0] + R[:, 0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
