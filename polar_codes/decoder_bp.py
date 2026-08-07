"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _element_update_left(left, right, alpha):
    upper = alpha * f_operation(right[1] + left[1], left[0])
    lower = alpha * f_operation(left[0], right[0]) + left[1]
    return upper, lower


def _element_update_right(left, right, alpha):
    upper = alpha * f_operation(right[1] + left[1], right[0])
    lower = alpha * f_operation(left[0], right[0]) + right[1]
    return upper, lower


def _bp_update_left(left_col, right_col, layer, alpha):
    N = len(left_col)
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_col[idx], left_col[idx + interval]])
            right_ele = np.array([right_col[idx], right_col[idx + interval]])
            up, lo = _element_update_left(left_ele, right_ele, alpha)
            value[idx] = up
            value[idx + interval] = lo
    return value


def _bp_update_right(left_col, right_col, layer, alpha):
    N = len(left_col)
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_col[idx], left_col[idx + interval]])
            right_ele = np.array([right_col[idx], right_col[idx + interval]])
            up, lo = _element_update_right(left_ele, right_ele, alpha)
            value[idx] = up
            value[idx + interval] = lo
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.LARGE = np.inf

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_perm = llr_ch[self.rev]

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_perm

        for i in range(N):
            if self.frozen_bits[i]:
                right_matrix[i, 0] = self.LARGE
            else:
                right_matrix[i, 0] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if u_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

        return u_hat, num_iters
