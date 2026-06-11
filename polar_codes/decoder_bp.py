"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * f_operation(a, b)


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_min_sum(right[1] + left[1], left[0], alpha)
    value[1] = _f_min_sum(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_min_sum(right[1] + left[1], right[0], alpha)
    value[1] = _f_min_sum(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([
                left_array[2 * i * interval + j],
                left_array[2 * i * interval + j + interval],
            ])
            right_ele = np.array([
                right_array[2 * i * interval + j],
                right_array[2 * i * interval + j + interval],
            ])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([
                left_array[2 * i * interval + j],
                left_array[2 * i * interval + j + interval],
            ])
            right_ele = np.array([
                right_array[2 * i * interval + j],
                right_array[2 * i * interval + j + interval],
            ])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = np.where(self.frozen_bits == 0)[0]
        self.LARGE = 1e9

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        information_pos = self.information_pos
        frozen_bit = 0

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr
        right_matrix[:, 0] = np.array([
            0 if i in information_pos else (1 - 2 * frozen_bit) * self.LARGE
            for i in range(N)
        ])

        num_iters = 0
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

            num_iters = it
            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
