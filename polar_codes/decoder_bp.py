"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _sign_with_zero_one
from encoder import polar_encode


def _ms_f(a, b, alpha):
    """min-sum f 运算，带缩放因子 alpha"""
    return (
        alpha
        * _sign_with_zero_one(a)
        * _sign_with_zero_one(b)
        * np.minimum(np.abs(a), np.abs(b))
    )


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = _ms_f(right[1] + left[1], left[0], alpha)
    value[1] = _ms_f(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = _ms_f(right[1] + left[1], right[0], alpha)
    value[1] = _ms_f(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, col_n, alpha):
    N = len(left_array)
    interval = 2 ** (col_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            get_value = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, col_n, alpha):
    N = len(left_array)
    interval = 2 ** (col_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            right_ele = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            get_value = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i, self.alpha)

            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, self.alpha)

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int32)
        return np.array_equal(x_hat, hard_ch)
