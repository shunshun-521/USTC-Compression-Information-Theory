"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（flooded scheduling）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        if N & (N - 1):
            raise ValueError("N must be power of 2")
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_ms(
                        R[i, j - 1] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = self._f_ms(R[i, j - 1], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 2 ** (j)
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_ms(
                        R[i + s, j + 1] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + s, j + 1]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
