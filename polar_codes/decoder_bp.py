"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum）。列 0 为信源端，列 n 为信道端。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        L[i + t, j - 1] = self._f_ms(
                            R[i + t, j] + L[i + t + s, j], L[i + t, j]
                        )
                        L[i + t + s, j - 1] = self._f_ms(
                            R[i + t, j], L[i + t, j]
                        ) + L[i + t + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        R[i + t, j + 1] = self._f_ms(
                            R[i + t + s, j] + L[i + t + s, j + 1], R[i + t, j]
                        )
                        R[i + t + s, j + 1] = self._f_ms(
                            R[i + t, j], L[i + t, j + 1]
                        ) + R[i + t + s, j]

            u_hat = self._hard_decision(L, R)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        return (total < 0).astype(int)
