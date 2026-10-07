"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i, j]: 从右向左；R[i, j]: 从左向右；i 为行索引 0..N-1，j 为层 0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        Li = i + k
                        Lis = i + k + step
                        L[Li, j - 1] = _minsum_f(
                            R[Li, j] + L[Lis, j],
                            L[Li, j],
                            self.alpha,
                        )
                        L[Lis, j - 1] = _minsum_f(
                            R[Li, j],
                            L[Li, j],
                            self.alpha,
                        ) + L[Lis, j]

            # 左到右更新 R
            for j in range(0, n):
                step = 2 ** (j)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        Li = i + k
                        Lis = i + k + step
                        R[Li, j + 1] = _minsum_f(
                            R[Lis, j] + L[Lis, j + 1],
                            R[Li, j],
                            self.alpha,
                        )
                        R[Lis, j + 1] = _minsum_f(
                            R[Li, j],
                            L[Li, j + 1],
                            self.alpha,
                        ) + R[Lis, j]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
