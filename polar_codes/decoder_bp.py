"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(0, n):
                span = 1 << j
                for block in range(0, N, 2 * span):
                    top = np.arange(block, block + span)
                    bot = top + span
                    R[top, j + 1] = self._f(R[bot, j] + L[bot, j + 1], R[top, j])
                    R[bot, j + 1] = self._f(R[top, j], L[top, j + 1]) + R[bot, j]

            for j in range(n - 1, -1, -1):
                span = 1 << j
                for block in range(0, N, 2 * span):
                    top = np.arange(block, block + span)
                    bot = top + span
                    L[top, j] = self._f(L[top, j + 1], L[bot, j + 1] + R[bot, j])
                    L[bot, j] = self._f(R[top, j], L[top, j + 1]) + L[bot, j + 1]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
