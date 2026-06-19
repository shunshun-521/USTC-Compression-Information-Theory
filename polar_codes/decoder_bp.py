"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        La = R[i, j + 1] + L[i + s, j + 1]
                        Lb = L[i, j + 1]
                        L[i, j] = _f_minsum(La, Lb, self.alpha)
                        L[i + s, j] = (
                            _f_minsum(R[i, j + 1], L[i, j + 1], self.alpha) + L[i + s, j + 1]
                        )

            for j in range(1, n):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j] = _f_minsum(
                            R[i + s, j] + L[i + s, j + 1],
                            R[i, j - 1],
                            self.alpha,
                        )
                        R[i + s, j] = (
                            _f_minsum(R[i, j - 1], L[i, j + 1], self.alpha)
                            + R[i + s, j - 1]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
