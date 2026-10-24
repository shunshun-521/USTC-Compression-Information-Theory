"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e7

    def _g(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]
        m = self.m
        N = self.N
        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)
        L[m, :] = llr_ch
        R[0, self.frozen_bits] = self.LARGE
        R[0, ~self.frozen_bits] = 0.0

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(m - 1, -1, -1):
                step = 1 << i
                for j in range(0, N, 2 * step):
                    for k in range(step):
                        a = j + k
                        b = j + k + step
                        L[i, a] = self._g(L[i + 1, a], L[i + 1, b] + R[i, b])
                        L[i, b] = self._g(L[i + 1, a], R[i, a]) + L[i + 1, b]

            for i in range(0, m):
                step = 1 << i
                for j in range(0, N, 2 * step):
                    for k in range(step):
                        a = j + k
                        b = j + k + step
                        R[i + 1, a] = self._g(R[i, a], L[i + 1, b] + R[i, b])
                        R[i + 1, b] = self._g(L[i + 1, a], R[i, a]) + R[i, b]

            u_hat = (L[0, :] < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(np.int8)
            inv = np.empty(N, dtype=np.int64)
            inv[self.br] = np.arange(N)
            if np.array_equal(x_hat, hard_x[inv]):
                break

        u_hat = (L[0, :] < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
