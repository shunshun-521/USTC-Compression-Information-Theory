"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import _reorder_channel_llr, f_operation


def _bp_element_left(left_l, right_l, right_r, left_r):
    """2x2 处理单元：左向 L 消息更新"""
    v0 = f_operation(right_r + left_r, left_l)  # alpha applied in f via caller
    v1 = f_operation(left_l, right_l) + left_r
    return v0, v1


def _bp_element_right(left_l, right_l, right_r, left_r):
    """2x2 处理单元：右向 R 消息更新"""
    v0 = f_operation(right_r + left_r, right_r)
    v1 = f_operation(left_l, right_l) + right_r
    return v0, v1


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _reorder_channel_llr(llr_orig, self.N)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        La = L[idx, j]
                        Lb = L[idx + s, j]
                        Ra = R[idx, j - 1]
                        Rb = R[idx + s, j]
                        L[idx, j - 1] = self._f_ms(Ra + Lb, La)
                        L[idx + s, j - 1] = self._f_ms(Ra, La) + Lb

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        La = L[idx, j + 1]
                        Lb = L[idx + s, j + 1]
                        Ra = R[idx, j]
                        Rb = R[idx + s, j]
                        R[idx, j + 1] = self._f_ms(Rb + Lb, Ra)
                        R[idx + s, j + 1] = self._f_ms(Ra, La) + Rb

            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_orig)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        posterior = L[:, 0] + R[:, 0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
