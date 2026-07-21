"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in self.frozen_idx:
            R[i, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j - 1] = _f_minsum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], self.alpha
                        )
                        L[idx_l, j - 1] = (
                            _f_minsum(R[idx_u, j], L[idx_u, j], self.alpha) + L[idx_l, j]
                        )

            for j in range(0, n):
                step = 2 ** (j)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j + 1] = _f_minsum(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j], self.alpha
                        )
                        R[idx_l, j + 1] = (
                            _f_minsum(R[idx_u, j], L[idx_u, j + 1], self.alpha) + R[idx_l, j]
                        )

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                if i in self.frozen_idx:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            if i in self.frozen_idx:
                u_hat[i] = 0

        return u_hat.astype(int), num_iters
