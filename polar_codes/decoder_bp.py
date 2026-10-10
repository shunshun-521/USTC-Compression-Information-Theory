"""
极化码 BP（置信传播）译码器
基于因子图（Abbas et al. min-sum 更新），含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.m
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for _ in range(1, self.max_iter + 1):
            num_iters = _

            for j in range(m - 1, -1, -1):
                pow2 = 1 << (m - 1 - j)
                for i in range(0, N, 2 * pow2):
                    i2 = i + pow2
                    L[i2, j] = L[i2, j + 1] + self._f_ms(L[i, j + 1], R[i, j])
                    L[i, j] = self._f_ms(
                        L[i, j + 1], L[i2, j + 1] + R[i2, j]
                    )

            for j in range(0, m):
                pow2 = 1 << j
                for i in range(0, N, 2 * pow2):
                    i2 = i + pow2
                    R[i, j + 1] = self._f_ms(R[i, j], L[i2, j + 1] + R[i2, j])
                    R[i2, j + 1] = R[i2, j] + self._f_ms(L[i, j + 1], R[i, j])

            total = L[:, 0] + R[:, 0]
            for i in range(N):
                u_hat[i] = 0 if total[i] >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        for i in range(N):
            u_hat[i] = 0 if total[i] >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
