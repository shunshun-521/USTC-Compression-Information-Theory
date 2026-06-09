"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr
from decoder_sc import f_operation, g_operation, _prepare_llr


def _element_update_left(left, right, alpha):
    v = np.zeros(2)
    v[0] = alpha * np.sign(right[1] + left[1]) * np.sign(left[0]) * min(abs(right[1] + left[1]), abs(left[0]))
    v[1] = f_operation(left[0], right[0]) * alpha + left[1]
    return v


def _element_update_right(left, right, alpha):
    v = np.zeros(2)
    v[0] = alpha * np.sign(right[1] + left[1]) * np.sign(right[0]) * min(abs(right[1] + left[1]), abs(right[0]))
    v[1] = f_operation(left[0], right[0]) * alpha + right[1]
    return v


def _bp_update_left(left_array, right_array, col, alpha):
    N = left_array.size
    interval = 2 ** (col - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, col, alpha):
    N = left_array.size
    interval = 2 ** (col - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        llr_internal = _prepare_llr(llr_ch)
        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_internal
        right_matrix[:, 0] = np.where(self.frozen_bits, self.LARGE, 0.0)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                return u_hat, it
            num_iters = it

        return u_hat, num_iters
