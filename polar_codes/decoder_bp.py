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

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch.astype(np.float64)
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li, Ri = i, i + s
                    L[Li, j + 1] = self._f_min_sum(R[Li, j] + L[Ri, j], L[Li, j])
                    L[Ri, j + 1] = self._f_min_sum(R[Li, j], L[Li, j]) + L[Ri, j]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li, Ri = i, i + s
                    R[Li, j + 1] = self._f_min_sum(R[Ri, j] + L[Ri, j + 1], R[Li, j])
                    R[Ri, j + 1] = self._f_min_sum(R[Li, j], L[Li, j + 1]) + R[Ri, j]

            num_iters = it + 1

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0

        return u_hat, num_iters
