"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation_minsum, _frozen_bool


def _minsum(a, b, alpha):
    return alpha * f_operation_minsum(a, b)


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        L[i + t, j - 1] = _minsum(
                            R[i + t, j - 1] + L[i + t + s, j],
                            L[i + t, j],
                            self.alpha,
                        )
                        L[i + t + s, j - 1] = _minsum(
                            R[i + t, j - 1], L[i + t, j], self.alpha
                        ) + L[i + t + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        R[i + t, j + 1] = _minsum(
                            R[i + t + s, j] + L[i + t + s, j + 1],
                            R[i + t, j],
                            self.alpha,
                        )
                        R[i + t + s, j + 1] = _minsum(
                            R[i + t, j], L[i + t, j + 1], self.alpha
                        ) + R[i + t + s, j]

            # 早停
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
