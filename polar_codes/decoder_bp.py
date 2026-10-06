"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        mapped = llr_ch[self.br]
        L[n, :] = mapped
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0, i1 = i + k, i + k + step
                        L[j - 1, i0] = self._f_ms(
                            R[j, i0] + L[j, i1], L[j, i0]
                        )
                        L[j - 1, i1] = self._f_ms(R[j, i0], L[j, i0]) + L[j, i1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0, i1 = i + k, i + k + step
                        R[j + 1, i0] = self._f_ms(
                            R[j, i1] + L[j + 1, i1], R[j, i0]
                        )
                        R[j + 1, i1] = self._f_ms(R[j, i0], L[j + 1, i0]) + R[j, i1]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
