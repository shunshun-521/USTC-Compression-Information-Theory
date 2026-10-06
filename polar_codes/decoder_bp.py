"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits) != 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr = np.asarray(llr_ch, dtype=np.float64).copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i : i + s, j - 1] = self._f_ms(
                        R[i : i + s, j] + L[i + s : i + 2 * s, j],
                        L[i : i + s, j],
                    )
                    L[i + s : i + 2 * s, j - 1] = self._f_ms(
                        R[i : i + s, j],
                        L[i : i + s, j],
                    ) + L[i + s : i + 2 * s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i : i + s, j + 1] = self._f_ms(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                        R[i : i + s, j],
                    )
                    R[i + s : i + 2 * s, j + 1] = self._f_ms(
                        R[i : i + s, j],
                        L[i : i + s, j + 1],
                    ) + R[i + s : i + 2 * s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat, num_iters
