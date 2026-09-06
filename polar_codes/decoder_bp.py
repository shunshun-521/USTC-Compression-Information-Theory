"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    s1 = np.sign(a) if np.sign(a) != 0 else 1
    s2 = np.sign(b) if np.sign(b) != 0 else 1
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    return np.array(
        [
            _f_min_sum(right[1] + left[1], left[0], alpha),
            _f_min_sum(left[0], right[0], alpha) + left[1],
        ]
    )


def _element_update_right(left, right, alpha):
    return np.array(
        [
            _f_min_sum(right[1] + left[1], right[0], alpha),
            _f_min_sum(left[0], right[0], alpha) + right[1],
        ]
    )


def _bp_update_left(left_array, right_array, stage, alpha):
    n = stage
    N = left_array.size
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    n = stage
    N = left_array.size
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch

        for i in range(N):
            if self.frozen_bits[i]:
                right_matrix[i, 0] = self.LARGE
            else:
                right_matrix[i, 0] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
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

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if u_llr[idx] >= 0 else 1

        return u_hat, num_iters
