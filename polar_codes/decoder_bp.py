"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(left[0]) * min(
        abs(right[1] + left[1]), abs(left[0])
    )
    value[1] = (
        alpha
        * np.sign(left[0])
        * np.sign(right[0])
        * min(abs(left[0]), abs(right[0]))
        + left[1]
    )
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(right[0]) * min(
        abs(right[1] + left[1]), abs(right[0])
    )
    value[1] = (
        alpha
        * np.sign(left[0])
        * np.sign(right[0])
        * min(abs(left[0]), abs(right[0]))
        + right[1]
    )
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
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
    N = left_array.size
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
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = list(np.where(~self.frozen_bits)[0])
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = np.where(
            self.frozen_bits, self.LARGE, 0.0
        )

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                    alpha,
                )

            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                    alpha,
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

        return u_hat, num_iters
