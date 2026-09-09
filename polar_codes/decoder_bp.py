"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    left = slice(i, i + step)
                    right = slice(i + step, i + (step << 1))
                    L[left, j - 1] = self._f_ms(
                        R[left, j] + L[right, j],
                        L[left, j],
                    )
                    L[right, j - 1] = self._f_ms(R[left, j], L[left, j]) + L[right, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    left = slice(i, i + step)
                    right = slice(i + step, i + (step << 1))
                    R[left, j + 1] = self._f_ms(
                        R[right, j] + L[right, j + 1],
                        R[left, j],
                    )
                    R[right, j + 1] = self._f_ms(R[left, j], L[left, j + 1]) + R[right, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
