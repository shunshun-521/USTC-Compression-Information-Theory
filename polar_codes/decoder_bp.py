"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        ncol = n + 2  # 列 0..n+1，信道 LLR 在列 n+1

        L = np.zeros((N, ncol), dtype=np.float64)
        R = np.zeros((N, ncol), dtype=np.float64)
        L[:, n + 1] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 右到左更新 L（j = n .. 1）
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        L[i + k, j] = self._f_ms(
                            R[i + k, j] + L[i + k + stride, j + 1],
                            L[i + k, j + 1],
                        )
                        L[i + k + stride, j] = (
                            self._f_ms(R[i + k, j], L[i + k, j + 1])
                            + L[i + k + stride, j + 1]
                        )

            # 左到右更新 R（j = 0 .. n-1）
            for j in range(0, n):
                stride = 1 << j
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        R[i + k, j + 1] = self._f_ms(
                            R[i + k + stride, j + 1] + L[i + k + stride, j + 1],
                            R[i + k, j],
                        )
                        R[i + k + stride, j + 1] = (
                            self._f_ms(R[i + k, j], L[i + k, j + 1])
                            + R[i + k + stride, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
