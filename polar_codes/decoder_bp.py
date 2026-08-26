"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.information_pos = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _element_update_left(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = self._minsum(right[1] + left[1], left[0])
        value[1] = self._minsum(left[0], right[0]) + left[1]
        return value

    def _element_update_right(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = self._minsum(right[1] + left[1], right[0])
        value[1] = self._minsum(left[0], right[0]) + right[1]
        return value

    def _bp_update_left(self, left_array, right_array, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for block in range(num):
            base = 2 * block * interval
            for j in range(interval):
                left_ele = np.array([left_array[base + j], left_array[base + j + interval]])
                right_ele = np.array([right_array[base + j], right_array[base + j + interval]])
                out = self._element_update_left(left_ele, right_ele)
                value[base + j] = out[0]
                value[base + j + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for block in range(num):
            base = 2 * block * interval
            for j in range(interval):
                left_ele = np.array([left_array[base + j], left_array[base + j + interval]])
                right_ele = np.array([right_array[base + j], right_array[base + j + interval]])
                out = self._element_update_right(left_ele, right_ele)
                value[base + j] = out[0]
                value[base + j + interval] = out[1]
        return value

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        br = bit_reversal_permutation(N)
        llr_ch = llr_ch[br]

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr_ch
        right[:, 0] = 0.0
        frozen_mask = np.ones(N, dtype=bool)
        frozen_mask[self.information_pos] = False
        right[frozen_mask, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                left[:, layer - 1] = self._bp_update_left(left[:, layer], right[:, layer - 1], layer)

            for layer in range(1, n + 1):
                right[:, layer] = self._bp_update_right(left[:, layer], right[:, layer - 1], layer)

            total = left[:, 0] + right[:, 0]
            u_hat[:] = 0
            u_hat[self.information_pos] = (total[self.information_pos] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = left[:, 0] + right[:, 0]
        u_hat[:] = 0
        u_hat[self.information_pos] = (total[self.information_pos] < 0).astype(int)
        return u_hat, num_iters
