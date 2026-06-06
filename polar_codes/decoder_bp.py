"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        s1 = np.sign(x)
        s2 = np.sign(y)
        s1 = np.where(s1 == 0, 1, s1)
        s2 = np.where(s2 == 0, 1, s2)
        return self.alpha * s1 * s2 * np.minimum(np.abs(x), np.abs(y))

    def _element_update_left(self, left, right):
        out = np.zeros(2, dtype=np.float64)
        out[0] = self._f_min_sum(right[1] + left[1], left[0])
        out[1] = self._f_min_sum(left[0], right[0]) + left[1]
        return out

    def _element_update_right(self, left, right):
        out = np.zeros(2, dtype=np.float64)
        out[0] = self._f_min_sum(right[1] + left[1], right[0])
        out[1] = self._f_min_sum(left[0], right[0]) + right[1]
        return out

    def _bp_update_left(self, left_array, right_array, layer):
        N = left_array.shape[0]
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_array[base], left_array[base + interval]])
                right_ele = np.array([right_array[base], right_array[base + interval]])
                upd = self._element_update_left(left_ele, right_ele)
                value[base] = upd[0]
                value[base + interval] = upd[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer):
        N = left_array.shape[0]
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_array[base], left_array[base + interval]])
                right_ele = np.array([right_array[base], right_array[base + interval]])
                upd = self._element_update_right(left_ele, right_ele)
                value[base] = upd[0]
                value[base + interval] = upd[1]
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for i in range(n):
                L[:, n - i - 1] = self._bp_update_left(L[:, n - i], R[:, n - i - 1], n - i)
            for i in range(n):
                R[:, i + 1] = self._bp_update_right(L[:, i + 1], R[:, i], i + 1)

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = np.where(llr_ch >= 0, 0, 1).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
