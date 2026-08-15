"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_boxplus, sc_decode
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
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f(self, a, b):
        return self.alpha * f_boxplus(a, b)

    def _layered_bp_step(self, L, R, llr_ch):
        """执行一次双向 min-sum 消息传递。"""
        N = self.N
        n = self.n

        L[:, n] = llr_ch
        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                L[i, j - 1] = self._f(
                    L[i, j] + R[i, j], L[i + s, j] + R[i + s, j]
                )
                L[i + s, j - 1] = self._f(R[i, j], L[i, j]) + L[
                    i + s, j
                ] + R[i + s, j]

        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                R[i, j + 1] = self._f(
                    R[i + s, j] + L[i + s, j + 1], R[i, j]
                )
                R[i + s, j + 1] = self._f(R[i, j], L[i, j + 1]) + R[
                    i + s, j
                ]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = sc_decode(llr_ch, self.frozen_bits)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            self._layered_bp_step(L, R, llr_ch)

            total = L[:, 0] + R[:, 0]
            total[self.frozen_idx] = self.LARGE
            u_bp = (total < 0).astype(int)
            u_bp[self.frozen_idx] = 0

            if self._early_stop(u_bp, llr_ch):
                return u_bp, num_iters

            u_hat = sc_decode(llr_ch + 0.25 * (L[:, 0] + R[:, 0]), self.frozen_bits)

        if not self._early_stop(u_hat, llr_ch):
            u_hat = sc_decode(llr_ch, self.frozen_bits)

        return u_hat, num_iters

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
