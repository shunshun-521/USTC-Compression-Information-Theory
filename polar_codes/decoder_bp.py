"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_minsum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _hard_decision(self, L, R):
        u = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            t = L[i, 0] + R[i, 0]
            u[i] = 0 if t >= 0 else 1
            if self.frozen_bits[i]:
                u[i] = 0
        return u

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        L[idx, j - 1] = _f_minsum(
                            R[idx, j] + L[s, j], L[idx, j], self.alpha
                        )
                        L[s, j - 1] = _f_minsum(R[idx, j], L[idx, j], self.alpha) + L[
                            s, j
                        ]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        R[idx, j + 1] = _f_minsum(
                            R[s, j] + L[s, j + 1], R[idx, j], self.alpha
                        )
                        R[s, j + 1] = _f_minsum(R[idx, j], L[idx, j + 1], self.alpha) + R[
                            s, j
                        ]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
