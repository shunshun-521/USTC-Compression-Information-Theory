"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for s in range(n - 1, -1, -1):
                stride = 1 << (s + 1)
                half = 1 << s
                for i in range(0, N, stride):
                    for j in range(half):
                        top = i + j
                        btm = i + j + half
                        L[top, s] = self._f_min_sum(
                            R[top, s + 1] + L[btm, s + 1], L[top, s + 1]
                        )
                        L[btm, s] = self._f_min_sum(
                            R[top, s + 1], L[top, s + 1]
                        ) + L[btm, s + 1]

            for s in range(n):
                stride = 1 << (s + 1)
                half = 1 << s
                for i in range(0, N, stride):
                    for j in range(half):
                        top = i + j
                        btm = i + j + half
                        R[top, s + 1] = self._f_min_sum(
                            R[btm, s] + L[btm, s + 1], R[top, s]
                        )
                        R[btm, s + 1] = self._f_min_sum(
                            R[top, s], L[top, s + 1]
                        ) + R[btm, s]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

        return u_hat, num_iters
