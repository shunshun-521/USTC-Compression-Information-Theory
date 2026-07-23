"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_update_left(self, left_col, right_col, stage):
        """从右向左更新 L 消息。"""
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_col[base], left_col[base + interval]])
                right_ele = np.array([right_col[base], right_col[base + interval]])
                out0 = self._f_min_sum(right_ele[1] + left_ele[1], left_ele[0])
                out1 = self._f_min_sum(left_ele[0], right_ele[0]) + left_ele[1]
                value[base] = out0
                value[base + interval] = out1
        return value

    def _bp_update_right(self, left_col, right_col, stage):
        """从左向右更新 R 消息。"""
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_col[base], left_col[base + interval]])
                right_ele = np.array([right_col[base], right_col[base + interval]])
                out0 = self._f_min_sum(right_ele[1] + left_ele[1], right_ele[0])
                out1 = self._f_min_sum(left_ele[0], right_ele[0]) + right_ele[1]
                value[base] = out0
                value[base + interval] = out1
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = self._bp_update_left(L[:, stage], R[:, stage - 1], stage)

            for stage in range(1, n + 1):
                R[:, stage] = self._bp_update_right(L[:, stage], R[:, stage - 1], stage)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
