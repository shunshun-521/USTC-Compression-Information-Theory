"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    L[i, j + 1] = self._minsum(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j + 1] = self._minsum(
                        R[i, j], L[i, j]
                    ) + L[i + step, j]

            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step * 2):
                    R[i, j + 1] = self._minsum(
                        R[i + step, j], L[i + step, j]
                    ) + R[i, j]
                    R[i + step, j + 1] = self._minsum(
                        R[i, j] + L[i, j], R[i + step, j]
                    )

            total = L[:, n] + R[:, n]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, n] + R[:, n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
