"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
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
        """min-sum 近似 f 运算，带 alpha 修正"""
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s * 2):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        L[idx_i, j - 1] = self._f_min_sum(
                            R[idx_i, j] + L[idx_is, j], L[idx_i, j]
                        )
                        L[idx_is, j - 1] = self._f_min_sum(
                            R[idx_i, j], L[idx_i, j]
                        ) + L[idx_is, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s * 2):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        R[idx_i, j + 1] = self._f_min_sum(
                            R[idx_is, j] + L[idx_is, j + 1], R[idx_i, j]
                        )
                        R[idx_is, j + 1] = self._f_min_sum(
                            R[idx_i, j], L[idx_i, j + 1]
                        ) + R[idx_is, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
