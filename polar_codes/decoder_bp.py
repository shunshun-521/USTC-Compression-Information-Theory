"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_f_transform, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = 1e6

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    La = R[i, j] + L[i, j + 1]
                    Lb = L[i + step, j + 1]
                    L[i, j] = self._f_ms(La, Lb)
                    Rb = R[i, j]
                    L[i + step, j] = self._f_ms(Rb, L[i, j + 1]) + L[i + step, j + 1]

            for j in range(n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Rb = R[i + step, j] + L[i + step, j + 1]
                    Ra = R[i, j - 1] if j > 0 else R[i, 0]
                    R[i, j] = self._f_ms(Rb, Ra)
                    R[i + step, j] = self._f_ms(Ra, L[i, j + 1]) + R[i + step, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_f_transform(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
