"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, _prepare_channel_llr


class BPDecoder:
    """BP 译码器（flooded scheduling，信道 LLR 比特倒序对齐）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool).reshape(-1)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        a = R[i + t, j - 1] + L[i + t + s, j]
                        b = L[i + t, j]
                        L[i + t, j - 1] = self._f_ms(a, b)
                        L[i + t + s, j - 1] = self._f_ms(R[i + t, j - 1], L[i + t, j]) + L[
                            i + t + s, j
                        ]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        R[i + t, j + 1] = self._f_ms(
                            R[i + t + s, j] + L[i + t + s, j + 1], R[i + t, j]
                        )
                        R[i + t + s, j + 1] = (
                            self._f_ms(R[i + t, j], L[i + t, j + 1]) + R[i + t + s, j]
                        )

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
