"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                block = 1 << s
                for i in range(0, N, 2 * block):
                    for k in range(block):
                        idx = i + k
                        L[idx, s] = _minsum_f(
                            R[idx, s + 1] + L[idx + block, s + 1],
                            L[idx, s + 1],
                            self.alpha,
                        )
                        L[idx + block, s] = (
                            _minsum_f(R[idx, s + 1], L[idx, s + 1], self.alpha)
                            + L[idx + block, s + 1]
                        )

            for s in range(n):
                block = 1 << s
                for i in range(0, N, 2 * block):
                    for k in range(block):
                        idx = i + k
                        R[idx, s + 1] = _minsum_f(
                            R[idx + block, s] + L[idx + block, s + 1],
                            R[idx, s],
                            self.alpha,
                        )
                        R[idx + block, s + 1] = (
                            _minsum_f(R[idx, s], L[idx, s + 1], self.alpha)
                            + R[idx + block, s]
                        )

            num_iters = it
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
