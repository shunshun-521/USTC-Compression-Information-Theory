"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation, _info_positions
from encoder import polar_encode


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * f_operation(right[1] + left[1], left[0])
    value[1] = f_operation(left[0], right[0]) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = alpha * f_operation(right[1] + left[1], right[0])
    value[1] = f_operation(left[0], right[0]) + right[1]
    return value


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_pos = _info_positions(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        for i in range(N):
            if i in self.info_pos:
                R[i, 0] = 0.0
            else:
                R[i, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i, self.alpha)
            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, self.alpha)

            post = L[:, 0] + R[:, 0]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_bits != 0] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        return u_hat, num_iters
