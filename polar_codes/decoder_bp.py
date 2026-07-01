"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from channel import hard_decision_llr
from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.rev]

        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = _minsum_f(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], self.alpha
                        )
                        L[idx_l, j - 1] = _minsum_f(
                            R[idx_u, j], L[idx_u, j], self.alpha
                        ) + L[idx_l, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j] = _minsum_f(
                            R[idx_l, j] + L[idx_l, j], R[idx_u, j - 1], self.alpha
                        )
                        R[idx_l, j] = _minsum_f(
                            R[idx_u, j - 1], L[idx_u, j], self.alpha
                        ) + R[idx_l, j]

            total = L[:, 0] + R[:, 0]
            u_internal = np.where(total >= 0, 0, 1).astype(int)
            u_internal[self.frozen_bits] = 0
            x_hat = polar_encode(u_internal)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_internal = np.where(total >= 0, 0, 1).astype(int)
            u_internal[self.frozen_bits] = 0

        return u_internal, num_iters
