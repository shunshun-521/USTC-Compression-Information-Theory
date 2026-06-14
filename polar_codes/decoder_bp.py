"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr

LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)

        left[:, n] = llr_ch[bit_reversal_permutation(N)]
        right[:, 0] = 0.0
        right[self.frozen_bits, 0] = LARGE

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        left[idx, j - 1] = self._f_ms(
                            right[idx, j] + left[idx + s, j],
                            left[idx, j],
                        )
                        left[idx + s, j - 1] = self._f_ms(
                            right[idx, j],
                            left[idx, j],
                        ) + left[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        right[idx, j + 1] = self._f_ms(
                            right[idx + s, j] + left[idx + s, j + 1],
                            right[idx, j],
                        )
                        right[idx + s, j + 1] = self._f_ms(
                            right[idx, j],
                            left[idx, j + 1],
                        ) + right[idx + s, j]

            total = left[:, 0] + right[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                return u_hat, it

        total = left[:, 0] + right[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
