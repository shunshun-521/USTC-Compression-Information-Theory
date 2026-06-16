"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e7

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _col(self, L, i, j):
        """列 j+1 在边界处退化为列 n"""
        col = min(j + 1, self.n)
        return L[i, col]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = self.LARGE

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = self._col(L, i, j)
                    Lis = self._col(L, i + s, j)
                    L[i, j - 1] = self._f_ms(R[i, j] + Lis, Li)
                    L[i + s, j - 1] = self._f_ms(R[i, j], Li) + Lis

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_ms(
                        R[i + s, j] + self._col(L, i + s, j), R[i, j]
                    )
                    R[i + s, j + 1] = (
                        self._f_ms(R[i, j], self._col(L, i + s, j)) + R[i + s, j]
                    )

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

            L[:, n] = llr_ch[self.br]

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
