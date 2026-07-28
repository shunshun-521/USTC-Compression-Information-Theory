"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = np.where(self.frozen_bits, self.LARGE, 0.0)

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 从左到右更新 R 消息
            for lam in range(n):
                s = 2 ** lam
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        i1 = i + j
                        i2 = i + j + s
                        R[i1, lam + 1] = self._f_min_sum(
                            R[i2, lam + 1] + L[i2, lam + 1],
                            R[i1, lam],
                        )
                        R[i2, lam + 1] = (
                            self._f_min_sum(R[i1, lam], L[i1, lam + 1])
                            + R[i2, lam]
                        )

            # 从右到左更新 L 消息
            for lam in range(n - 1, -1, -1):
                s = 2 ** lam
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        i1 = i + j
                        i2 = i + j + s
                        L[i1, lam] = self._f_min_sum(
                            L[i1, lam + 1],
                            R[i2, lam + 1] + L[i2, lam + 1],
                        )
                        L[i2, lam] = (
                            self._f_min_sum(L[i1, lam + 1], R[i1, lam])
                            + L[i2, lam + 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            x_hard = np.where(llr_ch >= 0, 0, 1)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
