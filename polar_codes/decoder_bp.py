"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        n = self.n
        N = self.N
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br].copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0 = i + t
                        i1 = i + t + step
                        L[i0, j] = self._f_ms(
                            R[i0, j + 1] + L[i1, j + 1], L[i0, j + 1]
                        )
                        L[i1, j] = self._f_ms(R[i0, j + 1], L[i0, j + 1]) + L[i1, j + 1]

            for j in range(n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0 = i + t
                        i1 = i + t + step
                        R[i0, j + 1] = self._f_ms(
                            R[i1, j + 1] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f_ms(R[i0, j], L[i1, j + 1]) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
