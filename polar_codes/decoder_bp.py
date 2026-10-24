"""
极化码 BP（置信传播）译码器：min-sum + 早停
"""
import numpy as np
from encoder import polar_encode


def _sgn(x):
    return np.where(x >= 0, 1.0, -1.0)


def _f_ms(a, b, alpha):
    return alpha * _sgn(a) * _sgn(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n，列 n 为信道）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        L[:, 0] = llr_ch.copy()
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = _f_ms(
                        R[i, j + 1] + L[i + s, j + 1], L[i, j + 1], alpha
                    )
                    L[i + s, j] = (
                        _f_ms(R[i, j + 1], L[i, j + 1], alpha) + L[i + s, j + 1]
                    )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_ms(
                        R[i + s, j + 1] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = (
                        _f_ms(R[i, j], L[i, j + 1], alpha) + R[i + s, j + 1]
                    )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_x):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
