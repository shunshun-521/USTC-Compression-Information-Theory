"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _element_update_left(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], left[0])
    value[1] = alpha * f_operation(left[0], right[0]) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], right[0])
    value[1] = alpha * f_operation(left[0], right[0]) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j],
                                 left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j],
                                  right_array[2 * i * interval + j + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j],
                                 left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j],
                                  right_array[2 * i * interval + j + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, llr_total):
        u_hat = (llr_total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch

        temp = np.zeros(N, dtype=np.float64)
        for i in range(N):
            temp[i] = 0.0 if i in self.info_idx else self.LARGE
        right_matrix[:, 0] = temp

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            u_hat = self._hard_decision(left_matrix[:, 0] + right_matrix[:, 0])
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._hard_decision(left_matrix[:, 0] + right_matrix[:, 0])
        return u_hat, num_iters
