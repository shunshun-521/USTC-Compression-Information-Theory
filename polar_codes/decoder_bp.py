"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _prepare_channel_llr, f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_work = _prepare_channel_llr(llr_natural)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_work
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    left = np.arange(block, block + step)
                    right = left + step
                    L[left, j - 1] = self._minsum_f(
                        R[left, j] + L[right, j],
                        L[left, j],
                    )
                    L[right, j - 1] = self._minsum_f(
                        R[left, j],
                        L[left, j],
                    ) + L[right, j]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    left = np.arange(block, block + step)
                    right = left + step
                    R[left, j + 1] = self._minsum_f(
                        R[right, j] + L[right, j + 1],
                        R[left, j],
                    )
                    R[right, j + 1] = self._minsum_f(
                        R[left, j],
                        L[left, j + 1],
                    ) + R[right, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
