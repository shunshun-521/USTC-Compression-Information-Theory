"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode, prepare_decoder_llr


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
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = prepare_decoder_llr(llr_ch)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = _f_min_sum(
                            R[i, j] + L[i + s, j], L[i, j], alpha
                        )
                        L[i + s, j - 1] = _f_min_sum(R[i, j], L[i, j], alpha) + L[
                            i + s, j
                        ]

            for j in range(0, n - 1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = _f_min_sum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j], alpha
                        )
                        R[i + s, j + 1] = _f_min_sum(R[i, j], L[i, j + 1], alpha) + R[
                            i + s, j
                        ]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat.astype(int), num_iters
