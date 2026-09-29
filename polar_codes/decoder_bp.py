"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, _generator_matrix


def _f_hf_sms(L1, L2, alpha=0.9375):
    s1 = 1 if L1 >= 0 else -1
    s2 = 1 if L2 >= 0 else -1
    return alpha * s1 * s2 * min(abs(L1), abs(L2))


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_hf_sms(right[1] + left[1], left[0], alpha)
    value[1] = _f_hf_sms(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_hf_sms(right[1] + left[1], right[0], alpha)
    value[1] = _f_hf_sms(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha):
    n = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(n / (interval * 2))
    value = np.zeros(n)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            get_value = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    n = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(n / (interval * 2))
    value = np.zeros(n)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            get_value = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


class BPDecoder:
    """BP 译码器（与 F^{\otimes n} 因子图一致）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._G = _generator_matrix(self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        information_pos = np.where(~self.frozen_bits)[0]

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch
        temp = np.zeros(N)
        temp[self.frozen_bits] = np.inf
        right_matrix[:, 0] = temp

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

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

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_d_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = (x_d_llr < 0).astype(int)
            x_g = (u_hat @ self._G) % 2
            if np.array_equal(x_g, x_d):
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_d_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
