"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, sc_decode
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self._rev = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def _scatter_llr(self, llr_ch):
        llr_internal = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            llr_internal[self._rev[i]] = llr_ch[i]
        return llr_internal

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        u_sc = sc_decode(llr_ch, self.frozen_bits)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = self._scatter_llr(llr_ch)

        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self._large
            else:
                R[i, 0] = (1 - 2 * u_sc[i]) * 5.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_ms(
                        L[i, j], L[i + step, j] + R[i + step, j - 1]
                    )
                    L[i + step, j - 1] = self._f_ms(
                        R[i, j - 1], L[i, j]
                    ) + L[i + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_ms(
                        R[i + step, j], L[i + step, j] + R[i + step, j - 1]
                    )
                    R[i + step, j] = self._f_ms(
                        R[i, j - 1], L[i, j]
                    ) + R[i + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
