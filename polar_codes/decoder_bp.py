"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j], R[i][j] for j in 0..n, i in 0..N-1
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        alpha = self.alpha
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1
            # Right to left: update L at columns n-1 .. 0
            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for ii in range(i, i + s):
                        L[ii, j] = _minsum_f(
                            R[ii, j] + L[ii + s, j + 1], L[ii, j + 1], alpha
                        )
                        L[ii + s, j] = (
                            _minsum_f(R[ii, j], L[ii, j + 1], alpha) + L[ii + s, j + 1]
                        )

            # Left to right: update R at columns 0 .. n-1
            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for ii in range(i, i + s):
                        R[ii, j + 1] = _minsum_f(
                            R[ii + s, j] + L[ii + s, j + 1], R[ii, j], alpha
                        )
                        R[ii + s, j + 1] = (
                            _minsum_f(R[ii, j], L[ii, j + 1], alpha) + R[ii + s, j]
                        )

            # Early stopping
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
