"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(self.br)

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_bits(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_br
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    La = R[i : i + s, j - 1] + L[i + s : i + 2 * s, j]
                    Lb = L[i : i + s, j]
                    L[i : i + s, j - 1] = self._minsum(La, Lb)
                    L[i + s : i + 2 * s, j - 1] = self._minsum(
                        R[i : i + s, j - 1], Lb
                    ) + L[i + s : i + 2 * s, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    Ra = R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1]
                    Rb = R[i : i + s, j - 1] if j > 0 else R[i : i + s, 0]
                    Lb = L[i : i + s, j + 1]
                    R[i : i + s, j] = self._minsum(Ra, Rb)
                    R[i + s : i + 2 * s, j] = self._minsum(Rb, Lb) + R[
                        i + s : i + 2 * s, j
                    ]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
