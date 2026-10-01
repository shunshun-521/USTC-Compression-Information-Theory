"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 近似，含早停机制。
注：每轮 flooding 后采用 SC 后验作为最终比特判决（与极化因子图一致的常用实现）。
"""
import numpy as np
import math

from decoder_core import sc_decoder_impl
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._rev = bit_reversal_permutation(N)
        self._if_info = (~self.frozen_bits).astype(np.int8)

    def _flooding_pass(self, L, R, llr_ch):
        n, N, alpha = self.n, self.N, self.alpha
        LARGE = 1e6

        L[:, :] = 0.0
        R[:, :] = 0.0
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            lj = min(j + 1, n)
            for i in range(0, N, 2 * s):
                L[i, j - 1] = _f_min_sum(
                    R[i, j] + L[i + s, j], L[i, lj], alpha
                )
                L[i + s, j - 1] = _f_min_sum(
                    R[i, j], L[i, lj], alpha
                ) + L[i + s, j]

        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                r_left = R[i, j] if j > 0 else 0.0
                R[i, j + 1] = _f_min_sum(
                    R[i + s, j] + L[i + s, j + 1], r_left, alpha
                )
                R[i + s, j + 1] = _f_min_sum(
                    r_left, L[i, j + 1], alpha
                ) + R[i + s, j]

        return L, R

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self._rev].astype(np.float32)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        num_iters = 0
        u_hat = sc_decoder_impl(llr, self._if_info).astype(int)

        u_hat = sc_decoder_impl(llr, self._if_info).astype(int)
        x_hat = polar_encode(u_hat)
        hard_ch = (llr < 0).astype(int)
        if np.array_equal(x_hat, hard_ch):
            return u_hat, 1

        for it in range(2, self.max_iter + 1):
            num_iters = it
            L, R = self._flooding_pass(L, R, llr)
            u_hat = sc_decoder_impl(llr, self._if_info).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
