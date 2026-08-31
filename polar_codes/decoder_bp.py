"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


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

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        # L[i][j]: left messages, R[i][j]: right messages
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # Right to left: update L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = _f_min_sum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], self.alpha
                        )
                        L[idx_l, j - 1] = (
                            _f_min_sum(R[idx_u, j], L[idx_u, j], self.alpha)
                            + L[idx_l, j]
                        )

            # Left to right: update R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j + 1] = _f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j], self.alpha
                        )
                        R[idx_l, j + 1] = (
                            _f_min_sum(R[idx_u, j], L[idx_u, j + 1], self.alpha)
                            + R[idx_l, j]
                        )

            # Decision
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            # Early stopping: compare in channel order
            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            inv_br = np.argsort(br)
            x_hard = hard_decision_llr(llr_ch)[inv_br]
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        return u_hat, num_iters
