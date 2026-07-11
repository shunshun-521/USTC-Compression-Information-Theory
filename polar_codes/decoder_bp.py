"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import build_generator_matrix


def _f_min_sum(a, b, alpha):
    s1 = np.sign(a)
    s2 = np.sign(b)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return alpha * s1 * s2 * min(abs(a), abs(b))


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


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.information_pos = list(self.info_indices)
        self.G = build_generator_matrix(N)

    def decode(self, llr_ch):
        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr
        right_matrix[:, 0] = np.array(
            [
                np.inf if i not in self.information_pos else 0.0
                for i in range(N)
            ]
        )

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
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            num_iters = it
            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_d_llr < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = (x_d_llr < 0).astype(int)
            x_g = (u_hat @ self.G) % 2
            if np.array_equal(x_g, x_d):
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_d_llr < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
