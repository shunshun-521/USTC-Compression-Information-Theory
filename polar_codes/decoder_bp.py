"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(l1, l2, alpha=0.9375):
    s1 = 1.0 if l1 >= 0 else -1.0
    s2 = 1.0 if l2 >= 0 else -1.0
    return alpha * s1 * s2 * min(abs(l1), abs(l2))


def _bp_update_left(left_array, right_array, left_array_n):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            le = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            re = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            value[2 * i * interval + j] = _f_min_sum(re[1] + le[1], le[0])
            value[2 * i * interval + j + interval] = _f_min_sum(le[0], re[0]) + le[1]
    return value


def _bp_update_right(left_array, right_array, left_array_n):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            le = np.array(
                [left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]]
            )
            re = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            value[2 * i * interval + j] = _f_min_sum(re[1] + le[1], re[0])
            value[2 * i * interval + j + interval] = _f_min_sum(le[0], re[0]) + re[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch

        inf_val = 1e9
        right_init = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if self.frozen_bits[i]:
                right_init[i] = -inf_val
            else:
                right_init[i] = 0.0
        right_matrix[:, 0] = right_init

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            x_enc = polar_encode(u_hat)
            if np.array_equal(x_enc, x_hard):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
