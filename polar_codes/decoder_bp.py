"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    sign_a = np.where(a >= 0, 1.0, -1.0)
    sign_b = np.where(b >= 0, 1.0, -1.0)
    return alpha * sign_a * sign_b * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, step << 1):
                    base = block
                    for i in range(step):
                        idx = base + i
                        idx2 = base + i + step
                        L[idx, j - 1] = _ms_f(
                            R[idx, j - 1] + L[idx2, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx2, j - 1] = _ms_f(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, step << 1):
                    base = block
                    for i in range(step):
                        idx = base + i
                        idx2 = base + i + step
                        R[idx, j + 1] = _ms_f(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx2, j + 1] = _ms_f(
                            R[idx, j],
                            L[idx, j + 1],
                            self.alpha,
                        ) + R[idx2, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_indices] = 0

        return u_hat, num_iters
