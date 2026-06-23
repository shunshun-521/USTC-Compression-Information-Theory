"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode

_LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.br_inv = np.empty(N, dtype=int)
        self.br_inv[self.br] = np.arange(N)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_layout = llr_ch[self.br_inv]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_layout
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = _LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    Li = R[i, j - 1] + L[i + step, j]
                    Lj = L[i, j]
                    Lki = L[i + step, j]
                    L[i, j - 1] = self._f_min_sum(Li, Lj)
                    L[i + step, j - 1] = self._f_min_sum(R[i, j - 1], Lj) + Lki

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Ri = R[i + step, j + 1] + L[i + step, j + 1]
                    Rj = R[i, j]
                    Lj = L[i, j + 1]
                    Rki = R[i + step, j + 1]
                    R[i, j + 1] = self._f_min_sum(Ri, Rj)
                    R[i + step, j + 1] = self._f_min_sum(Rj, Lj) + Rki

            total = L[:, 0] + R[:, 0]
            u_internal = (total < 0).astype(int)
            u_internal[self.frozen_bits] = 0
            u_hat = u_internal.copy()

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_internal = (total < 0).astype(int)
        u_internal[self.frozen_bits] = 0
        return u_internal.astype(int), num_iters
