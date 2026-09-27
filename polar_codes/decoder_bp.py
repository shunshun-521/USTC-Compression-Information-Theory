"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for phi in range(n - 1, -1, -1):
                step = 1 << phi
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        L[i + j, phi] = self._f_min_sum(
                            R[i + j, phi + 1] + L[i + j + step, phi + 1],
                            L[i + j, phi + 1],
                        )
                        L[i + j + step, phi] = self._f_min_sum(
                            R[i + j, phi + 1], L[i + j, phi + 1]
                        ) + L[i + j + step, phi + 1]

            for phi in range(n):
                step = 1 << phi
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        R[i + j, phi + 1] = self._f_min_sum(
                            R[i + j + step, phi + 1] + L[i + j + step, phi + 1],
                            R[i + j, phi],
                        )
                        R[i + j + step, phi + 1] = self._f_min_sum(
                            R[i + j, phi], L[i + j, phi + 1]
                        ) + R[i + j + step, phi]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total >= 0] = 0
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
