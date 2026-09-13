"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum(x, y, alpha):
    """min-sum 近似 f 运算，带修正因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N, n = self.N, self.n
        frozen = self.frozen_bits
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[frozen, n] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 从右到左更新 L（列 n 到 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j] = _minsum(R[i, j] + L[i + s, j - 1], L[i, j - 1], alpha)
                    L[i + s, j] = (
                        _minsum(R[i, j], L[i, j - 1], alpha) + L[i + s, j - 1]
                    )

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _minsum(
                        R[i + s, j] + L[i + s, j - 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = (
                        _minsum(R[i, j], L[i, j - 1], alpha) + R[i + s, j]
                    )

            # 早停检查
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if (frozen[i] or total >= 0) else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if (frozen[i] or total >= 0) else 1

        return u_hat, num_iters
