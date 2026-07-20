"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr

LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block = 1 << (s + 1)
                half = block // 2
                for j in range(0, N, block):
                    for k in range(half):
                        idx = j + k
                        idx2 = j + k + half
                        L[idx, s + 1] = _f_min_sum(
                            R[idx, s] + L[idx2, s], L[idx, s], self.alpha
                        )
                        L[idx2, s + 1] = (
                            _f_min_sum(R[idx, s], L[idx, s], self.alpha) + L[idx2, s]
                        )

            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block // 2
                for j in range(0, N, block):
                    for k in range(half):
                        idx = j + k
                        idx2 = j + k + half
                        R[idx, s + 1] = _f_min_sum(
                            R[idx2, s] + L[idx2, s + 1], R[idx, s], self.alpha
                        )
                        R[idx2, s + 1] = (
                            _f_min_sum(R[idx, s], L[idx, s + 1], self.alpha) + R[idx2, s]
                        )

            total = L[:, n] + R[:, n]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, n] + R[:, n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
