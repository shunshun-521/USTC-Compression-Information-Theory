"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        n = self.n
        N = self.N
        rev = bit_reversal_permutation(N)
        llr_br = np.asarray(llr_ch, dtype=np.float64)[rev]

        L = np.zeros((n + 1, N))
        R = np.zeros((n + 1, N))
        L[n] = llr_br
        R[0] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(n - 1, -1, -1):
                block = 1 << stage
                for i in range(0, N, 2 * block):
                    L[stage, i:i + block] = self._f_ms(
                        R[stage + 1, i:i + block] + L[stage + 1, i + block:i + 2 * block],
                        L[stage + 1, i:i + block]
                    )
                    L[stage, i + block:i + 2 * block] = (
                        self._f_ms(R[stage + 1, i:i + block], L[stage + 1, i:i + block])
                        + L[stage + 1, i + block:i + 2 * block]
                    )

            for stage in range(0, n):
                block = 1 << stage
                for i in range(0, N, 2 * block):
                    R[stage + 1, i:i + block] = self._f_ms(
                        R[stage + 1, i + block:i + 2 * block] + L[stage + 1, i + block:i + 2 * block],
                        R[stage, i:i + block]
                    )
                    R[stage + 1, i + block:i + 2 * block] = (
                        self._f_ms(R[stage, i:i + block], L[stage + 1, i:i + block])
                        + R[stage + 1, i + block:i + 2 * block]
                    )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, num_iters

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
