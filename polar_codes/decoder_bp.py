"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        LARGE = 1e6

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, 0] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j + 1] = self._f_min_sum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j])
                        L[idx_l, j + 1] = self._f_min_sum(
                            R[idx_u, j], L[idx_u, j]) + L[idx_l, j]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j] = self._f_min_sum(
                            R[idx_l, j + 1] + L[idx_l, j + 1], R[idx_u, j + 1])
                        R[idx_l, j] = self._f_min_sum(
                            R[idx_u, j + 1], L[idx_u, j + 1]) + R[idx_l, j + 1]

            total_llr = L[:, n] + R[:, n]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it + 1
                break

        total_llr = L[:, n] + R[:, n]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
