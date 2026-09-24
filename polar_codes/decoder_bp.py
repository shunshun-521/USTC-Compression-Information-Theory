"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n = self.n
        N = self.N
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_br, 0] = LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = (
                            _f_min_sum(R[idx, j], L[idx, j], self.alpha) + L[idx2, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _f_min_sum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], self.alpha) + R[idx2, j]
                        )

            total_llr = L[:, 0] + R[:, 0]
            u_hat_br = np.zeros(N, dtype=np.int8)
            for i in range(N):
                u_hat_br[i] = 0 if self.frozen_br[i] or total_llr[i] >= 0 else 1

            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[self.br] = u_hat_br
            x_hat = polar_encode(u_hat)
            hard_ch = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat_br = np.zeros(N, dtype=np.int8)
        for i in range(N):
            u_hat_br[i] = 0 if self.frozen_br[i] or total_llr[i] >= 0 else 1

        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[self.br] = u_hat_br
        return u_hat, num_iters
