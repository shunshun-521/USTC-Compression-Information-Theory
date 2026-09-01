"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i,j]: left messages, R[i,j]: right messages
        # j = 0..n (layer), i = 0..N-1
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(self.max_iter):
            num_iters = it + 1

            # Right to left: update L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._minsum(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                        )
                        L[idx + s, j - 1] = self._minsum(
                            R[idx, j - 1],
                            L[idx, j],
                        ) + L[idx + s, j]

            # Left to right: update R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._minsum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                        )
                        R[idx + s, j + 1] = self._minsum(
                            R[idx, j],
                            L[idx, j + 1],
                        ) + R[idx + s, j]

            # Early stopping
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
