"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._br = bit_reversal_permutation(N)

    def _reorder_llr(self, llr_ch):
        return np.asarray(llr_ch, dtype=np.float64)[self._br]

    def _inverse_reorder_bits(self, u_natural):
        """将自然顺序 u 转回信道顺序，用于与 llr_ch 比较"""
        u_ch = np.zeros_like(u_natural)
        u_ch[self._br] = u_natural
        return u_ch

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = self._reorder_llr(llr_ch)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i : i + s, j]
                    Lis = L[i + s : i + 2 * s, j]
                    Ri = R[i : i + s, j]
                    L[i : i + s, j - 1] = _f_min_sum(Ri + Lis, Li, alpha)
                    L[i + s : i + 2 * s, j - 1] = _f_min_sum(Ri, Li, alpha) + Lis

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i : i + s, j]
                    Ris = R[i + s : i + 2 * s, j]
                    Li = L[i : i + s, j + 1]
                    Lis = L[i + s : i + 2 * s, j + 1]
                    R[i : i + s, j + 1] = _f_min_sum(Ris + Lis, Ri, alpha)
                    R[i + s : i + 2 * s, j + 1] = _f_min_sum(Ri, Li, alpha) + Ris

            total = L[:, 0] + R[:, 0]
            u_natural = np.where(total >= 0, 0, 1).astype(int)
            u_natural[self.frozen_idx] = 0
            x_hat = polar_encode(u_natural)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                return u_natural, num_iters

        total = L[:, 0] + R[:, 0]
        u_natural = np.where(total >= 0, 0, 1).astype(int)
        u_natural[self.frozen_idx] = 0
        return u_natural, num_iters
