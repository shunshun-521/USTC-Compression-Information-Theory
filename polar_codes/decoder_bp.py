"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, _permute_channel_llrs
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        hard_ch = (np.asarray(llr_ch, dtype=np.float64) < 0).astype(int)
        llr_ch = _permute_channel_llrs(llr_ch, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    La = R[i, j] + L[i + step, j + 1]
                    Lb = L[i, j + 1]
                    L[i, j] = self._f_min_sum(La, Lb)

                    La2 = R[i, j]
                    Lb2 = L[i, j + 1]
                    L[i + step, j] = self._f_min_sum(La2, Lb2) + L[i + step, j + 1]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j + 1], R[i, j - 1]
                    )
                    R[i + step, j] = self._f_min_sum(
                        R[i, j - 1], L[i, j + 1]
                    ) + R[i + step, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
