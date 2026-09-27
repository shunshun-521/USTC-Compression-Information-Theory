"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_raw[self.rev]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        L[k, j - 1] = self._f_ms(
                            R[k, j - 1] + L[k + s, j], L[k, j]
                        )
                        L[k + s, j - 1] = self._f_ms(R[k, j - 1], L[k, j]) + L[k + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        R[k, j + 1] = self._f_ms(
                            R[k + s, j] + L[k + s, j + 1], R[k, j]
                        )
                        R[k + s, j + 1] = self._f_ms(R[k, j], L[k, j + 1]) + R[k + s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat.astype(int), num_iters
