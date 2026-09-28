"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
from __future__ import annotations

import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, stride << 1):
                    for r in range(stride):
                        i = block + r
                        s = stride
                        L[i, j - 1] = _f_min_sum(R[i, j] + L[i + s, j], L[i, j], self.alpha)
                        L[i + s, j - 1] = _f_min_sum(R[i, j], L[i, j], self.alpha) + L[i + s, j]

            for j in range(1, n + 1):
                stride = 1 << (j - 1)
                for block in range(0, N, stride << 1):
                    for r in range(stride):
                        i = block + r
                        s = stride
                        R[i, j] = _f_min_sum(
                            R[i + s, j] + L[i + s, j], R[i, j - 1], self.alpha
                        )
                        R[i + s, j] = _f_min_sum(R[i, j - 1], L[i, j], self.alpha) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            num_iters = self.max_iter

        return u_hat, num_iters
