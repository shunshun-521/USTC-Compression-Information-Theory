"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation, _frozen_to_info_pos
from encoder import polar_encode


def _bp_element_left(left, right, alpha):
    v = np.zeros(2)
    v[0] = alpha * f_operation(right[1] + left[1], left[0])
    v[1] = alpha * f_operation(left[0], right[0]) + left[1]
    return v


def _bp_element_right(left, right, alpha):
    v = np.zeros(2)
    v[0] = alpha * f_operation(right[1] + left[1], right[0])
    v[1] = alpha * f_operation(left[0], right[0]) + right[1]
    return v


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            re = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            out = _bp_element_left(le, re, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            re = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            out = _bp_element_right(le, re, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = _frozen_to_info_pos(self.frozen_bits)
        self.info_set = set(int(i) for i in self.info_indices)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        LARGE = 1e8

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        for i in range(N):
            R[i, 0] = 0.0 if i in self.info_set else LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(
                    L[:, n - i], R[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            u_llr = L[:, 0] + R[:, 0]
            u_hat = np.array([0 if u_llr[i] >= 0 else 1 for i in range(N)])
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1
        return u_hat, num_iters
