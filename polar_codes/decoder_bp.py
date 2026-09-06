"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_from_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


def _element_update_left(left, right, alpha):
    return np.array([
        _f_min_sum(right[1] + left[1], left[0], alpha),
        _f_min_sum(left[0], right[0], alpha) + left[1],
    ])


def _element_update_right(left, right, alpha):
    return np.array([
        _f_min_sum(right[1] + left[1], right[0], alpha),
        _f_min_sum(left[0], right[0], alpha) + right[1],
    ])


def _bp_update_left(left_array, right_array, stage, alpha):
    n = stage
    N = left_array.shape[0]
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = left_array.copy()
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    n = stage
    N = left_array.shape[0]
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = right_array.copy()
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e100

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                    self.alpha,
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                    self.alpha,
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_indices] = (total[self.info_indices] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_from_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_indices] = (total[self.info_indices] < 0).astype(int)
        return u_hat, num_iters
