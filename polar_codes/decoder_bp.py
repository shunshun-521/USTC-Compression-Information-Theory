"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from decoder_sc import _reorder_channel_llrs


def bp_f(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _reorder_channel_llrs(llr_ch_orig)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for i in range(0, N, 2 * stride):
                    L[i : i + stride, j - 1] = bp_f(
                        R[i : i + stride, j] + L[i + stride : i + 2 * stride, j],
                        L[i : i + stride, j],
                        self.alpha,
                    )
                    L[i + stride : i + 2 * stride, j - 1] = bp_f(
                        R[i : i + stride, j],
                        L[i : i + stride, j],
                        self.alpha,
                    ) + L[i + stride : i + 2 * stride, j]

            for j in range(0, n):
                stride = 1 << j
                for i in range(0, N, 2 * stride):
                    R[i : i + stride, j + 1] = bp_f(
                        R[i + stride : i + 2 * stride, j] + L[i + stride : i + 2 * stride, j + 1],
                        R[i : i + stride, j],
                        self.alpha,
                    )
                    R[i + stride : i + 2 * stride, j + 1] = bp_f(
                        R[i : i + stride, j],
                        L[i : i + stride, j + 1],
                        self.alpha,
                    ) + R[i + stride : i + 2 * stride, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
