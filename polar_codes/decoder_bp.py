"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 n+1 列，列 0 为信源端，列 n 为信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        ch = llr_ch[rev]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左：更新 L 消息（列 n-1 .. 0）
            for j in range(self.n - 1, -1, -1):
                s = 1 << j
                for i in range(0, self.N, 2 * s):
                    for t in range(s):
                        L[i + t, j] = self._f_ms(
                            R[i + t, j + 1] + L[i + t + s, j + 1], L[i + t, j + 1]
                        )
                        L[i + t + s, j] = self._f_ms(R[i + t, j + 1], L[i + t, j + 1]) + L[
                            i + t + s, j + 1
                        ]

            # 左到右：更新 R 消息（列 1 .. n）
            for j in range(1, self.n + 1):
                s = 1 << (j - 1)
                for i in range(0, self.N, 2 * s):
                    for t in range(s):
                        R[i + t, j] = self._f_ms(
                            R[i + t + s, j - 1] + L[i + t + s, j], R[i + t, j - 1]
                        )
                        R[i + t + s, j] = self._f_ms(R[i + t, j - 1], L[i + t, j]) + R[
                            i + t + s, j - 1
                        ]

            for i in range(self.N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(self.N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
