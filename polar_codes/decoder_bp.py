"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.rev]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_br
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        i2 = i + s
                        L[i, j - 1] = self._minsum_f(R[i, j] + L[i2, j], L[i, j])
                        L[i2, j - 1] = self._minsum_f(R[i, j], L[i, j]) + L[i2, j]

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        i2 = i + s
                        R[i, j + 1] = self._minsum_f(R[i2, j] + L[i2, j + 1], R[i, j])
                        R[i2, j + 1] = self._minsum_f(R[i, j], L[i, j + 1]) + R[i2, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
