"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（因子图 stages 0..n，stage n 为信道端）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                for block in range(0, N, 2 * stride):
                    left = block
                    right = block + stride
                    for offset in range(stride):
                        i = left + offset
                        j = right + offset
                        L[stage, i] = self._f(
                            R[stage, i] + L[stage + 1, j], L[stage + 1, i]
                        )
                        L[stage, j] = (
                            self._f(R[stage, i], L[stage + 1, i])
                            + L[stage + 1, j]
                        )

            for stage in range(n):
                stride = 1 << stage
                for block in range(0, N, 2 * stride):
                    left = block
                    right = block + stride
                    for offset in range(stride):
                        i = left + offset
                        j = right + offset
                        R[stage + 1, i] = self._f(
                            R[stage + 1, j] + L[stage + 1, j], R[stage, i]
                        )
                        R[stage + 1, j] = (
                            self._f(R[stage, i], L[stage + 1, i]) + R[stage + 1, j]
                        )

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[0, i] + R[0, i]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat[self.rev], (llr < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[0, i] + R[0, i]) >= 0 else 1
            )

        return u_hat, num_iters
