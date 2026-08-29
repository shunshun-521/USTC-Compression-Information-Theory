"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_transform, bit_reversal_permutation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（min-sum，含早停）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        if x == 0:
            return y
        if y == 0:
            return x
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = self._f_ms(
                        R[i + step, j] + L[i + step, j + 1], R[i, j]
                    )
                    R[i + step, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[
                        i + step, j
                    ]

            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    L[i, j] = self._f_ms(
                        R[i, j + 1] + L[i + step, j + 1], L[i, j + 1]
                    )
                    L[i + step, j] = self._f_ms(R[i, j + 1], L[i, j + 1]) + L[
                        i + step, j + 1
                    ]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_transform(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
