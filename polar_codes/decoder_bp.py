"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for i in range(0, N, stride << 1):
                    for k in range(i, i + stride):
                        L[k, j - 1] = self._f_min_sum(
                            R[k, j] + L[k + stride, j],
                            L[k, j],
                        )
                        L[k + stride, j - 1] = self._f_min_sum(
                            R[k, j],
                            L[k, j],
                        ) + L[k + stride, j]

            for j in range(1, n + 1):
                stride = 1 << (j - 1)
                for i in range(0, N, stride << 1):
                    for k in range(i, i + stride):
                        R[k, j] = self._f_min_sum(
                            R[k + stride, j] + L[k + stride, j],
                            R[k, j - 1],
                        )
                        R[k + stride, j] = self._f_min_sum(
                            R[k, j - 1],
                            L[k, j],
                        ) + R[k + stride, j]

            u_hat = np.zeros(N, dtype=np.int8)
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
