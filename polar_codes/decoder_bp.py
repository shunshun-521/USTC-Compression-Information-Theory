"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation_minsum
from encoder import polar_encode
from channel import hard_decision_llr

LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation_minsum(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for r in range(step):
                        idx = i + r
                        idx2 = idx + step
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx2, j], L[idx, j]
                        )
                        L[idx2, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for r in range(step):
                        idx = i + r
                        idx2 = idx + step
                        R[idx, j + 1] = self._f_ms(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j]
                        )
                        R[idx2, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[idx2, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
