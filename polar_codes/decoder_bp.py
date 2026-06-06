"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr, reorder_llr_for_decode
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算，带修正因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = reorder_llr_for_decode(llr_natural, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i : i + s, j]
                    Lip1 = L[i : i + s, j + 1]
                    Lis_p1 = L[i + s : i + 2 * s, j + 1]

                    L[i : i + s, j] = _f_min_sum(Ri + Lis_p1, Lip1, self.alpha)
                    L[i + s : i + 2 * s, j] = _f_min_sum(Ri, Lip1, self.alpha) + Lis_p1

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ris = R[i + s : i + 2 * s, j]
                    Lis_p1 = L[i + s : i + 2 * s, j + 1]
                    Rim1 = R[i : i + s, j]
                    Lip1 = L[i : i + s, j + 1]

                    R[i : i + s, j + 1] = _f_min_sum(Ris + Lis_p1, Rim1, self.alpha)
                    R[i + s : i + 2 * s, j + 1] = _f_min_sum(Rim1, Lip1, self.alpha) + Ris

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_natural)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                return u_hat, num_iters

            num_iters = it

        return u_hat, num_iters
