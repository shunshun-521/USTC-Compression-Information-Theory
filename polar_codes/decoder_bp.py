"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[self.frozen_indices, n] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx, idx2 = i + k, i + k + s
                        L[idx, j + 1] = _minsum_f(
                            R[idx, j] + L[idx2, j], L[idx, j], alpha
                        )
                        L[idx2, j + 1] = (
                            _minsum_f(R[idx, j], L[idx, j], alpha) + L[idx2, j]
                        )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx, idx2 = i + k, i + k + s
                        R[idx, j - 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j], R[idx, j - 1], alpha
                        )
                        R[idx2, j - 1] = (
                            _minsum_f(R[idx, j - 1], L[idx, j], alpha) + R[idx2, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat[:] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat[:] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_indices] = 0

        return u_hat, num_iters
