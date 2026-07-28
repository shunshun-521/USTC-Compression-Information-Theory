"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _element_update_left(left, right, alpha=0.9375):
    s1 = np.sign(left[1] + right[1])
    s2 = np.sign(left[0])
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    v0 = alpha * s1 * s2 * min(abs(left[1] + right[1]), abs(left[0]))
    s3 = np.sign(left[0])
    s4 = np.sign(right[0])
    if s3 == 0:
        s3 = 1
    if s4 == 0:
        s4 = 1
    v1 = alpha * s3 * s4 * min(abs(left[0]), abs(right[0])) + left[1]
    return np.array([v0, v1])


def _element_update_right(left, right, alpha=0.9375):
    s1 = np.sign(right[1] + left[1])
    s2 = np.sign(right[0])
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    v0 = alpha * s1 * s2 * min(abs(right[1] + left[1]), abs(right[0]))
    s3 = np.sign(left[0])
    s4 = np.sign(right[0])
    if s3 == 0:
        s3 = 1
    if s4 == 0:
        s4 = 1
    v1 = alpha * s3 * s4 * min(abs(left[0]), abs(right[0])) + right[1]
    return np.array([v0, v1])


def _bp_update_left(left_array, right_array, left_array_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, left_array_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = N // (interval * 2)
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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_set = set(np.where(~self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = np.array([
            0.0 if i in self.info_set else self.LARGE for i in range(N)
        ])

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if u_llr[i] >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(np.int8)
            x_hat = polar_encode(u_hat)

            num_iters = it
            if np.array_equal(x_hat, x_hard):
                break

        u_hat[self.frozen_bits] = 0
        return u_hat.copy(), num_iters
