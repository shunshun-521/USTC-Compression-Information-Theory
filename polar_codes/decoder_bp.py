"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                block = stride << 1
                for base in range(0, N, block):
                    for k in range(stride):
                        i = base + k
                        j = i + stride
                        L[i, stage] = _f_min_sum(
                            L[i, stage + 1], L[j, stage + 1] + R[j, stage], self.alpha
                        )
                        L[j, stage] = _f_min_sum(
                            R[i, stage], L[i, stage + 1], self.alpha
                        ) + L[j, stage + 1]

            for stage in range(n):
                stride = 1 << stage
                block = stride << 1
                for base in range(0, N, block):
                    for k in range(stride):
                        i = base + k
                        j = i + stride
                        R[i, stage + 1] = _f_min_sum(
                            R[j, stage] + L[j, stage + 1], R[i, stage], self.alpha
                        )
                        R[j, stage + 1] = _f_min_sum(
                            R[i, stage], L[i, stage + 1], self.alpha
                        ) + R[j, stage]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
