"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha
        br = self.br

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _ms_f(R[i, j] + L[i + s, j], L[i, j + 1], alpha)
                    L[i + s, j - 1] = _ms_f(R[i, j], L[i, j + 1], alpha) + L[i + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _ms_f(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = _ms_f(R[i, j], L[i, j + 1], alpha) + R[i + s, j]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                tot = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if tot >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            tot = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if tot >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0
        return u_hat, num_iters
