"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（按层更新的因子图消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Ri = i + k + s
                        L[Li, j - 1] = _f_min_sum(R[Li, j] + L[Ri, j], L[Li, j], self.alpha)
                        L[Ri, j - 1] = _f_min_sum(R[Li, j], L[Li, j], self.alpha) + L[Ri, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Ri = i + k + s
                        R[Li, j + 1] = _f_min_sum(R[Ri, j] + L[Ri, j + 1], R[Li, j], self.alpha)
                        R[Ri, j + 1] = _f_min_sum(R[Li, j], L[Li, j + 1], self.alpha) + R[Ri, j]

            u_hat = (L[:, 0] + R[:, 0]) < 0
            u_hat = u_hat.astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
