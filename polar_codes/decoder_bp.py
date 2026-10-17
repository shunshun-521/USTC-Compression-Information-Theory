"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _init_messages(self, llr_ch, fixed_u=None):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE
        if fixed_u is not None:
            fixed_u = np.asarray(fixed_u, dtype=np.int32)
            for i in range(N):
                if fixed_u[i] >= 0:
                    R[i, 0] = self.LARGE if fixed_u[i] == 0 else -self.LARGE
        return L, R

    def posterior_llr(self, llr_ch, fixed_u=None):
        """单次 BP 迭代后的后验 LLR（L+R，列 0）。"""
        L, R = self._init_messages(llr_ch, fixed_u)
        n = self.n
        N = self.N
        for j in range(n, 0, -1):
            s = 2 ** (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    L[i + k, j - 1] = self._f_ms(
                        R[i + k, j - 1] + L[i + k + s, j],
                        L[i + k, j],
                    )
                    L[i + k + s, j - 1] = self._f_ms(
                        R[i + k, j - 1],
                        L[i + k, j],
                    ) + L[i + k + s, j]
        for j in range(0, n):
            s = 2 ** j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    R[i + k, j + 1] = self._f_ms(
                        R[i + k + s, j + 1] + L[i + k + s, j + 1],
                        R[i + k, j],
                    )
                    R[i + k + s, j + 1] = self._f_ms(
                        R[i + k, j],
                        L[i + k, j + 1],
                    ) + R[i + k + s, j]
        return L[:, 0] + R[:, 0]

    def decode(self, llr_ch, fixed_u=None):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L, R = self._init_messages(llr_ch, fixed_u)

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j - 1] = self._f_ms(
                            R[i + k, j - 1] + L[i + k + s, j],
                            L[i + k, j],
                        )
                        L[i + k + s, j - 1] = self._f_ms(
                            R[i + k, j - 1],
                            L[i + k, j],
                        ) + L[i + k + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 2 ** (j + 1 - 1)  # 2^j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j + 1] = self._f_ms(
                            R[i + k + s, j + 1] + L[i + k + s, j + 1],
                            R[i + k, j],
                        )
                        R[i + k + s, j + 1] = self._f_ms(
                            R[i + k, j],
                            L[i + k, j + 1],
                        ) + R[i + k + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int32)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
