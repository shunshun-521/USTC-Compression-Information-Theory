"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _ms_boxplus(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_indices:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for c in range(n - 1, -1, -1):
                stride = 1 << c
                for i in range(0, N, 2 * stride):
                    for t in range(stride):
                        a = i + t
                        b = i + t + stride
                        L[a, c] = _ms_boxplus(R[a, c + 1] + L[b, c + 1], L[a, c + 1], self.alpha)
                        L[b, c] = _ms_boxplus(R[a, c + 1], L[a, c + 1], self.alpha) + L[b, c + 1]

            for c in range(1, n + 1):
                stride = 1 << (c - 1)
                for i in range(0, N, 2 * stride):
                    for t in range(stride):
                        a = i + t
                        b = i + t + stride
                        R[a, c] = _ms_boxplus(R[b, c] + L[b, c], R[a, c - 1], self.alpha)
                        R[b, c] = _ms_boxplus(R[a, c - 1], L[a, c], self.alpha) + R[b, c]

            for i in range(N):
                u_hat[i] = 0 if i in self.frozen_indices else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if i in self.frozen_indices else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

        return u_hat, num_iters
