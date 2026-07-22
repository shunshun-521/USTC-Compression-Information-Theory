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
    BP 译码器（分层因子图，min-sum 近似）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _decide(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        brp = bit_reversal_permutation(self.N)
        llr_work = llr_ch[brp].copy()
        x_hard = hard_decision_llr(llr_ch)

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_work
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for s in range(n, 0, -1):
                stride = 1 << (s - 1)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        u = i + j
                        d = i + j + stride
                        L[u, s - 1] = self._f(R[u, s] + L[d, s], L[u, s])
                        L[d, s - 1] = self._f(R[u, s], L[u, s]) + L[d, s]

            L[:, n] = llr_work
            R[self.frozen_bits == 1, 0] = self.large

            for s in range(0, n):
                stride = 1 << s
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        u = i + j
                        d = i + j + stride
                        R[u, s + 1] = self._f(R[d, s] + L[d, s + 1], R[u, s])
                        R[d, s + 1] = self._f(R[u, s], L[u, s + 1]) + R[d, s]

            u_hat = self._decide(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, x_hard):
                break

        return u_hat, num_iters
