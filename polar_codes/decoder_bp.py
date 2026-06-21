"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import inverse_bit_reversal_permutation, polar_encode


def _bp_f(x, y, alpha):
    """BP min-sum f 函数"""
    sign_x = np.sign(x)
    sign_y = np.sign(y)
    sign_x = np.where(sign_x == 0, 1, sign_x)
    sign_y = np.where(sign_y == 0, 1, sign_y)
    return alpha * sign_x * sign_y * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.inv_br = inverse_bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _left_update(self, L, R, stage):
        """stage: 0..n-1，块大小 2^{stage+1}"""
        block = 1 << (stage + 1)
        half = block >> 1
        col_out = stage
        col_in = stage + 1
        for base in range(0, self.N, block):
            for k in range(half):
                i = base + k
                j = i + half
                L[i, col_out] = _bp_f(
                    L[i, col_in], L[j, col_in] + R[j, col_out], self.alpha
                )
                L[j, col_out] = _bp_f(
                    R[i, col_out], L[i, col_in], self.alpha
                ) + L[j, col_in]

    def _right_update(self, L, R, stage):
        block = 1 << (stage + 1)
        half = block >> 1
        col_out = stage + 1
        col_in = stage
        for base in range(0, self.N, block):
            for k in range(half):
                i = base + k
                j = i + half
                R[i, col_out] = _bp_f(
                    R[i, col_in], L[j, col_out] + R[j, col_in], self.alpha
                )
                R[j, col_out] = _bp_f(
                    R[i, col_in], L[i, col_out], self.alpha
                ) + R[j, col_in]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.inv_br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        stage_orders = [
            list(range(n - 1, -1, -1)),
            list(range(n)),
        ]

        for it in range(1, self.max_iter + 1):
            num_iters = it
            order = stage_orders[it % 2]

            if it % 2 == 1:
                for stage in order:
                    self._left_update(L, R, stage)
            else:
                for stage in order:
                    self._right_update(L, R, stage)

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
