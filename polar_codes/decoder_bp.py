"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


LARGE = 1e6


def _minsum_f(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        L[idx1, j - 1] = _minsum_f(
                            R[idx1, j] + L[idx2, j], L[idx1, j], alpha
                        )
                        L[idx2, j - 1] = _minsum_f(R[idx1, j], L[idx1, j], alpha) + L[
                            idx2, j
                        ]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        R[idx1, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx1, j], alpha
                        )
                        R[idx2, j + 1] = _minsum_f(
                            R[idx1, j], L[idx1, j + 1], alpha
                        ) + R[idx2, j]

            num_iters = it

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
