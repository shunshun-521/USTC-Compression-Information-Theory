"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation, _frozen_bits_to_info_pos
from encoder import polar_encode, build_generator_matrix


def _bp_element_update_left(left, right, alpha):
    """BP 左向消息更新（单个 2x2 PE）"""
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], left[0])
    value[1] = alpha * f_operation(left[0], right[0]) + left[1]
    return value


def _bp_element_update_right(left, right, alpha):
    """BP 右向消息更新（单个 2x2 PE）"""
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], right[0])
    value[1] = alpha * f_operation(left[0], right[0]) + right[1]
    return value


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _bp_element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = _frozen_bits_to_info_pos(frozen_bits)
        self.frozen_set = set(range(N)) - set(self.information_pos)
        self.G = build_generator_matrix(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e6

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        for i in range(N):
            right_matrix[i, 0] = 0.0 if i in self.information_pos else LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            num_iters = it
            llr_total = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.array([0 if llr_total[i] >= 0 else 1 for i in range(N)], dtype=int)
            for idx in self.frozen_set:
                u_hat[idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        llr_total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.array([0 if llr_total[i] >= 0 else 1 for i in range(N)], dtype=int)
        for idx in self.frozen_set:
            u_hat[idx] = 0
        return u_hat, num_iters
