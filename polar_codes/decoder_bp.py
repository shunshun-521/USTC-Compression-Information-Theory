"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_scaled(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li, Ri = i, i + s
                    L[Li, j] = self._f_scaled(
                        R[Li, j] + L[Ri, j + 1], L[Li, j + 1]
                    )
                    L[Ri, j] = self._f_scaled(R[Li, j], L[Li, j + 1]) + L[Ri, j + 1]

            # 左到右更新 R（列 1 到 n）
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li, Ri = i, i + s
                    R[Li, j] = self._f_scaled(
                        R[Ri, j] + L[Ri, j], R[Li, j - 1]
                    )
                    R[Ri, j] = self._f_scaled(R[Li, j - 1], L[Li, j]) + R[Ri, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
