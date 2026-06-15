"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE
        R[n, :] = 0.0

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for d in range(n - 1, -1, -1):
                stride = 1 << d
                for block in range(0, N, 2 * stride):
                    for i in range(block, block + stride):
                        j = i + stride
                        L[d, i] = self._f(R[d + 1, i] + L[d + 1, j], L[d + 1, i])
                        L[d, j] = self._f(R[d + 1, i], L[d + 1, i]) + L[d + 1, j]

            for d in range(1, n + 1):
                stride = 1 << (d - 1)
                for block in range(0, N, 2 * stride):
                    for i in range(block, block + stride):
                        j = i + stride
                        R[d, i] = self._f(R[d, j] + L[d, j], R[d - 1, i])
                        R[d, j] = self._f(R[d - 1, i], L[d, i]) + R[d, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

        return u_hat, num_iters
