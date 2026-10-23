"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.where(a >= 0, 1.0, -1.0)
    sb = np.where(b >= 0, 1.0, -1.0)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j]: left (check-to-bit) messages, R[i][j]: right messages
        # j = 0 .. n (column), i = 0 .. N-1 row index at each stage
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # Right to left: update L at columns n-1 .. 0
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        i2 = i + s
                        L[i, j] = _f_min_sum(
                            R[i, j] + L[i2, j + 1], L[i, j + 1], self.alpha
                        )
                        L[i2, j] = _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + L[
                            i2, j + 1
                        ]

            # Left to right: update R at columns 0 .. n-1
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        i2 = i + s
                        R[i, j + 1] = _f_min_sum(
                            R[i2, j] + L[i2, j + 1], R[i, j], self.alpha
                        )
                        R[i2, j + 1] = (
                            _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[i2, j]
                        )

            # Early stopping
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
