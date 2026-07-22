"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    left = slice(i, i + step)
                    right = slice(i + step, i + 2 * step)
                    L[left, j - 1] = self._f_min_sum(
                        R[left, j] + L[right, j],
                        L[left, j],
                    )
                    L[right, j - 1] = self._f_min_sum(
                        R[left, j],
                        L[left, j],
                    ) + L[right, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    left = slice(i, i + step)
                    right = slice(i + step, i + 2 * step)
                    R[left, j + 1] = self._f_min_sum(
                        R[right, j] + L[right, j + 1],
                        R[left, j],
                    )
                    R[right, j + 1] = self._f_min_sum(
                        R[left, j],
                        L[left, j + 1],
                    ) + R[right, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
