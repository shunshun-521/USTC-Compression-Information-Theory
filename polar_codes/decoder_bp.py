"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图消息传递）"""

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
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for l in range(n - 1, -1, -1):
                step = 1 << l
                for i in range(0, N, 2 * step):
                    j = i + step
                    L[i, l] = _f_min_sum(
                        L[i, l + 1],
                        R[j, l] + L[j, l + 1],
                        self.alpha,
                    )
                    L[j, l] = _f_min_sum(L[i, l + 1], R[i, l], self.alpha) + L[j, l + 1]

            for l in range(0, n):
                step = 1 << l
                for i in range(0, N, 2 * step):
                    j = i + step
                    R[i, l + 1] = _f_min_sum(
                        R[i, l],
                        L[j, l + 1] + R[j, l + 1],
                        self.alpha,
                    )
                    R[j, l + 1] = _f_min_sum(R[i, l], L[i, l + 1], self.alpha) + R[j, l + 1]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if total >= 0 else 1)

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if total >= 0 else 1)

        return u_hat, num_iters
