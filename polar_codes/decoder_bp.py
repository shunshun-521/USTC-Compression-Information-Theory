"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层因子图）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j], R[i][j]: i in 0..N-1, j in 0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # Right to left: update L at columns n-1 .. 0
            for j in range(n - 1, -1, -1):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    i2 = i + step
                    L[i, j] = _ms_f(
                        R[i, j] + L[i2, j + 1], L[i, j + 1], self.alpha
                    )
                    L[i2, j] = _ms_f(R[i, j], L[i, j + 1], self.alpha) + L[i2, j + 1]

            # Left to right: update R at columns 1 .. n
            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    i2 = i + step
                    R[i, j] = _ms_f(
                        R[i2, j] + L[i2, j], R[i, j - 1], self.alpha
                    )
                    R[i2, j] = _ms_f(R[i, j - 1], L[i, j], self.alpha) + R[i2, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
