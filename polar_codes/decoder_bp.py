"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        if N & (N - 1):
            raise ValueError("N must be power of 2")
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, step << 1):
                    for i in range(block, block + step):
                        i2 = i + step
                        L[i, j - 1] = self._f_ms(R[i, j] + L[i2, j], L[i, j])
                        L[i2, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i2, j]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, step << 1):
                    for i in range(block, block + step):
                        i2 = i + step
                        R[i, j + 1] = self._f_ms(R[i2, j] + L[i2, j + 1], R[i, j])
                        R[i2, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i2, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat
