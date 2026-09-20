"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


def _f_min_sum(a, b, alpha):
    s1 = np.sign(a) if a != 0 else 1.0
    s2 = np.sign(b) if b != 0 else 1.0
    return alpha * s1 * s2 * min(abs(a), abs(b))


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


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
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


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
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
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_pos = set(np.where(~self.frozen_bits)[0])

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr

        frozen_val = 0
        right_matrix[:, 0] = [
            (1 - 2 * frozen_val) * self.LARGE if i not in self.info_pos else 0.0
            for i in range(N)
        ]

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, alpha
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (y_llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
