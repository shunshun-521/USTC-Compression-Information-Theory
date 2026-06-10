"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_minsum
from encoder import polar_encode

_LARGE = 1e6


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

    def _minsum_f(self, a, b):
        return self.alpha * f_minsum(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for lam in range(n - 1, -1, -1):
                block = 1 << lam
                for phi in range(0, N, 2 * block):
                    for omega in range(block):
                        top = phi + omega
                        btm = top + block
                        L[top, lam] = self._minsum_f(L[top, lam + 1], L[btm, lam + 1])
                        L[btm, lam] = (
                            self._minsum_f(R[top, lam + 1], L[top, lam + 1])
                            + L[btm, lam + 1]
                        )

            for lam in range(n):
                block = 1 << lam
                for phi in range(0, N, 2 * block):
                    for omega in range(block):
                        top = phi + omega
                        btm = top + block
                        R[top, lam + 1] = (
                            self._minsum_f(R[btm, lam], L[btm, lam + 1]) + R[top, lam]
                        )
                        R[btm, lam + 1] = (
                            self._minsum_f(R[top, lam], L[btm, lam + 1]) + R[btm, lam]
                        )

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
