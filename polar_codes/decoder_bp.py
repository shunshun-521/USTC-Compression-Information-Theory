"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """极化码 BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        for it in range(self.max_iter):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j] = _f_min_sum(
                            R[idx, j + 1] + L[idx + s, j + 1],
                            L[idx, j + 1],
                            self.alpha,
                        )
                        L[idx + s, j] = (
                            _f_min_sum(R[idx, j + 1], L[idx, j + 1], self.alpha)
                            + L[idx + s, j + 1]
                        )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = _f_min_sum(
                            R[idx + s, j] + L[idx + s, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx + s, j] = (
                            _f_min_sum(R[idx, j - 1], L[idx, j], self.alpha)
                            + R[idx + s, j]
                        )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                )

            rev = bit_reversal_permutation(N)
            hard_channel = (llr_ch[rev] < 0).astype(int)
            if np.array_equal(polar_encode(u_hat), hard_channel):
                return u_hat, it + 1

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            )
        return u_hat, self.max_iter
