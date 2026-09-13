"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


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
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._init_messages()

    def _init_messages(self):
        n, N = self.n, self.N
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.R = np.zeros((N, n + 1), dtype=np.float64)

    def _hard_decision_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha
        large = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = _f_min_sum(
                            R[idx_u, j] + L[idx_l, j],
                            L[idx_u, j],
                            alpha,
                        )
                        L[idx_l, j - 1] = (
                            _f_min_sum(R[idx_u, j], L[idx_u, j], alpha)
                            + L[idx_l, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j + 1] = _f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1],
                            R[idx_u, j],
                            alpha,
                        )
                        R[idx_l, j + 1] = (
                            _f_min_sum(R[idx_u, j], L[idx_u, j + 1], alpha)
                            + R[idx_l, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = self._hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
