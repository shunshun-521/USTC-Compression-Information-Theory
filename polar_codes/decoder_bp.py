"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from channel import hard_decision_llr
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_update_left(self, left_col, right_col, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_col[idx], left_col[idx + interval]])
                right_ele = np.array([right_col[idx], right_col[idx + interval]])
                out0 = self._f_ms(right_ele[1] + left_ele[1], left_ele[0])
                out1 = self._f_ms(left_ele[0], right_ele[0]) + left_ele[1]
                value[idx] = out0
                value[idx + interval] = out1
        return value

    def _bp_update_right(self, left_col, right_col, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_col[idx], left_col[idx + interval]])
                right_ele = np.array([right_col[idx], right_col[idx + interval]])
                out0 = self._f_ms(right_ele[1] + left_ele[1], right_ele[0])
                out1 = self._f_ms(left_ele[0], right_ele[0]) + right_ele[1]
                value[idx] = out0
                value[idx + interval] = out1
        return value

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        br = bit_reversal_permutation(N)
        y_llr = np.asarray(llr_ch, dtype=np.float64)[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = y_llr

        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.large
            else:
                R[i, 0] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = self._bp_update_left(L[:, n - i], R[:, n - i - 1], n - i)
            for i in range(n):
                R[:, i + 1] = self._bp_update_right(L[:, i + 1], R[:, i], i + 1)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
