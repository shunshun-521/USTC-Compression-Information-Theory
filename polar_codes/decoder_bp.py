"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode


def _bp_element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * f_operation(right[1] + left[1], left[0])
    value[1] = alpha * f_operation(left[0], right[0]) + left[1]
    return value


def _bp_element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * f_operation(right[1] + left[1], right[0])
    value[1] = alpha * f_operation(left[0], right[0]) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _bp_element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 2 ** (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _bp_element_update_right(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器，含 min-sum 近似与早停。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = _prepare_channel_llr(llr_ch, self.N)

        L = np.zeros((self.N, self.n + 1))
        R = np.zeros((self.N, self.n + 1))
        L[:, self.n] = y_llr

        for i in range(self.N):
            if self.frozen_bits[i] == 1:
                R[i, 0] = self._large
            else:
                R[i, 0] = 0.0

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(self.n):
                L[:, self.n - i - 1] = _bp_update_left(
                    L[:, self.n - i], R[:, self.n - i - 1], self.n - i, self.alpha
                )

            for i in range(self.n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            u_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                if self.frozen_bits[i] == 1:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if u_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (y_llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i] == 1:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

        return u_hat, num_iters
