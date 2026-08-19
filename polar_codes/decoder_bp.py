"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b))


def _element_update_left(left, right):
    return np.array([
        _f_min_sum(right[1] + left[1], left[0]),
        _f_min_sum(left[0], right[0]) + left[1],
    ])


def _element_update_right(left, right):
    return np.array([
        _f_min_sum(right[1] + left[1], right[0]),
        _f_min_sum(left[0], right[0]) + right[1],
    ])


def _bp_update_left(left_array, right_array, layer_n):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            lo = 2 * i * interval + j
            hi = lo + interval
            left_ele = np.array([left_array[lo], left_array[hi]])
            right_ele = np.array([right_array[lo], right_array[hi]])
            out = _element_update_left(left_ele, right_ele)
            value[lo] = out[0]
            value[hi] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            lo = 2 * i * interval + j
            hi = lo + interval
            left_ele = np.array([left_array[lo], left_array[hi]])
            right_ele = np.array([right_array[lo], right_array[hi]])
            out = _element_update_right(left_ele, right_ele)
            value[lo] = out[0]
            value[hi] = out[1]
    return value


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e10

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or u_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] or u_llr[i] >= 0 else 1

        return u_hat, num_iters
