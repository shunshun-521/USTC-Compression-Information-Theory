"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _frozen_to_info_pos, f_operation
from encoder import polar_encode


def _element_update_left(left, right, alpha):
    """BP 左向消息更新（2 节点单元）"""
    value = np.zeros(2)
    value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(left[0]) * min(
        abs(right[1] + left[1]), abs(left[0])
    )
    value[1] = alpha * np.sign(left[0]) * np.sign(right[0]) * min(
        abs(left[0]), abs(right[0])
    ) + left[1]
    return value


def _element_update_right(left, right, alpha):
    """BP 右向消息更新（2 节点单元）"""
    value = np.zeros(2)
    value[0] = alpha * np.sign(right[1] + left[1]) * np.sign(right[0]) * min(
        abs(right[1] + left[1]), abs(right[0])
    )
    value[1] = alpha * np.sign(left[0]) * np.sign(right[0]) * min(
        abs(left[0]), abs(right[0])
    ) + right[1]
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
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = _frozen_to_info_pos(self.frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch

        frozen_bit = 0
        right_matrix[:, 0] = [
            (1 - 2 * frozen_bit) * self.LARGE
            if i not in self.information_pos else 0.0
            for i in range(N)
        ]

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1],
                    n - i, alpha,
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, alpha,
                )

            num_iters = it
            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

        return u_hat, num_iters
