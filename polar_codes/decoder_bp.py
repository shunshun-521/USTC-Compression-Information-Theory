"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 min-sum，按规范列索引实现）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e8

    def _f_minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        for i in range(N):
            R[i, 0] = self.large if i in self.frozen_set else 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            # 从右到左更新 L（列 n-1 .. 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    i2 = i + s
                    L[i, j] = self._f_minsum(R[i, j] + L[i2, j + 1], L[i, j + 1])
                    L[i2, j] = self._f_minsum(R[i, j], L[i, j + 1]) + L[i2, j + 1]

            # 从左到右更新 R（列 0 .. n-1）
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    i2 = i + s
                    R[i, j + 1] = self._f_minsum(R[i2, j] + L[i2, j + 1], R[i, j])
                    R[i2, j + 1] = self._f_minsum(R[i, j], L[i, j + 1]) + R[i2, j]

            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break
            num_iters = it + 1

        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
