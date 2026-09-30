"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


def _f_minsum(a, b, alpha=0.9375):
    sa = np.sign(a)
    sb = np.sign(b)
    if sa == 0:
        sa = 1
    if sb == 0:
        sb = 1
    return alpha * sa * sb * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_minsum(right[1] + left[1], left[0], alpha)
    value[1] = _f_minsum(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_minsum(right[1] + left[1], right[0], alpha)
    value[1] = _f_minsum(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, left_array_n, alpha):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [
                    right_array[2 * i * interval + j],
                    right_array[2 * i * interval + j + interval],
                ]
            )
            get_value = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, left_array_n, alpha):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [
                    right_array[2 * i * interval + j],
                    right_array[2 * i * interval + j + interval],
                ]
            )
            get_value = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_pos = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e8

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = llr_ch.copy()

        left = np.zeros((self.N, self.n + 1))
        right = np.zeros((self.N, self.n + 1))
        left[:, self.n] = y_llr

        frozen_bit = self.frozen_bits.copy()
        temp = (1 - 2 * frozen_bit) * self._large
        right[:, 0] = np.array(
            [temp[i] if i not in self.info_pos else 0.0 for i in range(self.N)]
        )

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(self.n):
                left[:, self.n - i - 1] = _bp_update_left(
                    left[:, self.n - i],
                    right[:, self.n - i - 1],
                    self.n - i,
                    self.alpha,
                )
            for i in range(self.n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )

            u_llr = left[:, 0] + right[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hard = (llr_ch < 0).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_llr = left[:, 0] + right[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
