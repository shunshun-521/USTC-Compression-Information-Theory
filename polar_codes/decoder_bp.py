"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, _prepare_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return float(alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b)))


def _bp_update_left(left_array, right_array, stage):
    N = left_array.size
    interval = 2 ** (stage - 1)
    value = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            value[idx] = _f_min_sum(right_ele[1] + left_ele[1], left_ele[0], 0.9375)
            value[idx + interval] = _f_min_sum(left_ele[0], right_ele[0], 0.9375) + left_ele[1]
    return value


def _bp_update_right(left_array, right_array, stage):
    N = left_array.size
    interval = 2 ** (stage - 1)
    value = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            value[idx] = _f_min_sum(right_ele[1] + left_ele[1], right_ele[0], 0.9375)
            value[idx + interval] = _f_min_sum(left_ele[0], right_ele[0], 0.9375) + right_ele[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        N = self.N
        n = self.n
        y_llr = _prepare_llr(llr_ch, N)

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = y_llr
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            total_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
