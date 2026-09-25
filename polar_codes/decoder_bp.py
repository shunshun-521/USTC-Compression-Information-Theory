"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e10

    def _f_minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                stride = 1 << s
                block = 2 * stride
                for base in range(0, N, block):
                    for j in range(base, base + stride):
                        L[s, j] = self._f_minsum(
                            R[s, j] + L[s + 1, j + stride], L[s + 1, j]
                        )
                        L[s, j + stride] = (
                            self._f_minsum(R[s, j], L[s + 1, j]) + L[s + 1, j + stride]
                        )

            for s in range(n):
                stride = 1 << s
                block = 2 * stride
                for base in range(0, N, block):
                    for j in range(base, base + stride):
                        R[s + 1, j] = self._f_minsum(
                            R[s + 1, j + stride] + L[s + 1, j + stride], R[s, j]
                        )
                        R[s + 1, j + stride] = (
                            self._f_minsum(R[s, j], L[s + 1, j]) + R[s + 1, j + stride]
                        )

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
