"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                block = 1 << (layer - 1)
                stride = 1 << layer
                for i in range(0, N, stride):
                    for b in range(block):
                        top = i + b
                        bot = i + b + block
                        L[top, layer - 1] = self._f(
                            R[top, layer] + L[bot, layer],
                            L[top, layer],
                        )
                        L[bot, layer - 1] = self._f(
                            R[top, layer],
                            L[top, layer],
                        ) + L[bot, layer]

            for layer in range(0, n):
                block = 1 << layer
                stride = 1 << (layer + 1)
                for i in range(0, N, stride):
                    for b in range(block):
                        top = i + b
                        bot = i + b + block
                        R[top, layer + 1] = self._f(
                            R[bot, layer] + L[bot, layer + 1],
                            R[top, layer],
                        )
                        R[bot, layer + 1] = self._f(
                            R[top, layer],
                            L[top, layer + 1],
                        ) + R[bot, layer]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
