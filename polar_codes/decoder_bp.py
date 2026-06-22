"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _ms_f(x, y, alpha):
    """min-sum f 函数，带归一化因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, alpha = self.n, self.N, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        R_ij = R[i, j]
                        L_i_j = L[i, j]
                        L_is_j = L[i + step, j]

                        L[i, j - 1] = _ms_f(R_ij + L_is_j, L_i_j, alpha)
                        L[i + step, j - 1] = _ms_f(R_ij, L_i_j, alpha) + L_is_j

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        R_is_j = R[i + step, j]
                        L_is_j1 = L[i + step, j + 1]
                        R_i_j = R[i, j]
                        L_i_j1 = L[i, j + 1]

                        R[i, j + 1] = _ms_f(R_is_j + L_is_j1, R_i_j, alpha)
                        R[i + step, j + 1] = _ms_f(R_i_j, L_i_j1, alpha) + R_is_j

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
