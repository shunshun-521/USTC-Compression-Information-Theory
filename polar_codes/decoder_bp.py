"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, reorder_channel_llr
from encoder import polar_encode, polar_generator_matrix


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left_array[base], left_array[base + interval]
            r0, r1 = right_array[base], right_array[base + interval]
            value[base] = alpha * f_operation(r1 + l1, l0)
            value[base + interval] = alpha * f_operation(l0, r0) + l1
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left_array[base], left_array[base + interval]
            r0, r1 = right_array[base], right_array[base + interval]
            value[base] = alpha * f_operation(r1 + l1, r0)
            value[base + interval] = alpha * f_operation(r0, l0) + r1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.information_pos = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self._G = polar_generator_matrix(N)

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        y_llr = reorder_channel_llr(llr_natural)
        N, n = self.N, self.n

        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = y_llr
        right[:, 0] = 0.0
        right[:, 0] = np.where(
            np.isin(np.arange(N), self.information_pos),
            0.0,
            self.large,
        )

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )
            num_iters = it

            total = left[:, 0] + right[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = left[:, 0] + right[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
