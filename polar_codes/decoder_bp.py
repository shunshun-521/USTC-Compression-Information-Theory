"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha=0.9375):
    """min-sum 近似的 box-plus 运算"""
    if a == 0.0:
        return b
    if b == 0.0:
        return a
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
  列 0：信源端；列 n：信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L 消息（列 n-1 到 0）
            for c in range(n - 1, -1, -1):
                s = 1 << c
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        L[Li, c] = _boxplus_minsum(
                            R[Li, c] + L[Li_s, c + 1],
                            L[Li, c + 1],
                            self.alpha,
                        )
                        L[Li_s, c] = (
                            _boxplus_minsum(R[Li, c], L[Li, c + 1], self.alpha)
                            + L[Li_s, c + 1]
                        )

            # 从左到右更新 R 消息（列 1 到 n）
            for c in range(1, n + 1):
                s = 1 << (c - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        R[Li, c] = _boxplus_minsum(
                            R[Li_s, c] + L[Li_s, c],
                            R[Li, c - 1],
                            self.alpha,
                        )
                        R[Li_s, c] = (
                            _boxplus_minsum(R[Li, c - 1], L[Li, c], self.alpha)
                            + R[Li_s, c]
                        )

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total_llr < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total_llr < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
