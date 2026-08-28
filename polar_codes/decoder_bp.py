"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                stride = 2 ** j
                for i in range(0, N, 2 * stride):
                    L[i, j] = self._f_ms(
                        R[i, j + 1] + L[i + stride, j + 1], L[i, j + 1]
                    )
                    L[i + stride, j] = self._f_ms(
                        R[i, j + 1], L[i, j + 1]
                    ) + L[i + stride, j + 1]

            for j in range(0, n):
                stride = 2 ** j
                for i in range(0, N, 2 * stride):
                    R[i, j + 1] = self._f_ms(
                        R[i + stride, j] + L[i + stride, j + 1], R[i, j]
                    )
                    R[i + stride, j + 1] = self._f_ms(
                        R[i, j], L[i, j + 1]
                    ) + R[i + stride, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
