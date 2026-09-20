"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch.astype(np.float64)
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for s in range(n):
                bs = 2 ** (s + 1)
                br = bs // 2
                for p in range(0, N, bs):
                    for k in range(br):
                        j = p + k
                        L[j, s + 1] = self._minsum(
                            R[j, s] + L[j + br, s], L[j, s]
                        )
                        L[j + br, s + 1] = self._minsum(
                            R[j, s], L[j, s]
                        ) + L[j + br, s]

            for s in range(n):
                bs = 2 ** (s + 1)
                br = bs // 2
                for p in range(0, N, bs):
                    for k in range(br):
                        j = p + k
                        R[j, s + 1] = self._minsum(
                            R[j + br, s] + L[j + br, s + 1], R[j, s]
                        )
                        R[j + br, s + 1] = self._minsum(
                            R[j, s], L[j, s + 1]
                        ) + R[j + br, s]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, n] + R[i, n]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, n] + R[i, n]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
