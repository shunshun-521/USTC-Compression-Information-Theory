"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """返回：(u_hat, num_iters)"""
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        z = llr_nat[self.rev]
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = z
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for layer in range(n - 1, -1, -1):
                block = 2 ** (layer + 1)
                half = block // 2
                for i in range(0, N, block):
                    for j in range(half):
                        L[layer, i + j] = self._f_min_sum(
                            L[layer + 1, i + j],
                            L[layer + 1, i + j + half] + R[layer + 1, i + j],
                        )
                        L[layer, i + j + half] = (
                            self._f_min_sum(R[layer + 1, i + j], L[layer + 1, i + j])
                            + L[layer + 1, i + j + half]
                        )

            for layer in range(n):
                block = 2 ** (layer + 1)
                half = block // 2
                for i in range(0, N, block):
                    for j in range(half):
                        R[layer + 1, i + j] = self._f_min_sum(
                            R[layer, i + j + half] + L[layer + 1, i + j + half],
                            R[layer, i + j],
                        )
                        R[layer + 1, i + j + half] = (
                            self._f_min_sum(R[layer, i + j], L[layer + 1, i + j])
                            + R[layer, i + j + half]
                        )

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
