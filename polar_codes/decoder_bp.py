"""
极化码 BP（置信传播）译码器
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        L[idx, j] = self._f_ms(
                            R[idx, j + 1] + L[idx + step, j + 1], L[idx, j + 1]
                        )
                        L[idx + step, j] = self._f_ms(R[idx, j + 1], L[idx, j + 1]) + L[
                            idx + step, j + 1
                        ]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        R[idx, j + 1] = self._f_ms(
                            R[idx + step, j + 1] + L[idx + step, j + 1], R[idx, j]
                        )
                        R[idx + step, j + 1] = self._f_ms(R[idx, j + 1], L[idx, j + 1]) + R[
                            idx + step, j + 1
                        ]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
        return u_hat.astype(int), num_iters
