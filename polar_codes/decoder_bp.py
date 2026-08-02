"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _element_update_left(self, left, right):
        return np.array([
            self._minsum_f(right[1] + left[1], left[0]),
            self._minsum_f(left[0], right[0]) + left[1],
        ])

    def _element_update_right(self, left, right):
        return np.array([
            self._minsum_f(right[1] + left[1], right[0]),
            self._minsum_f(left[0], right[0]) + right[1],
        ])

    def _bp_update_left(self, left_array, right_array, layer):
        N = len(left_array)
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = left_array.copy()
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]])
                right_ele = np.array([right_array[idx], right_array[idx + interval]])
                get_value = self._element_update_left(left_ele, right_ele)
                value[idx] = get_value[0]
                value[idx + interval] = get_value[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer):
        N = len(left_array)
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = right_array.copy()
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]])
                right_ele = np.array([right_array[idx], right_array[idx + interval]])
                get_value = self._element_update_right(left_ele, right_ele)
                value[idx] = get_value[0]
                value[idx + interval] = get_value[1]
        return value

    def decode(self, llr_ch):
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)

        left_array = np.zeros(N)
        right_array = llr.copy()
        left_array[self.frozen_bits] = self.LARGE

        for num_iters in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                left_array = self._bp_update_left(left_array, right_array, layer)
            for layer in range(1, n + 1):
                right_array = self._bp_update_right(left_array, right_array, layer)

            total = left_array + right_array
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if self._early_stop(u_hat, llr_ch):
                return u_hat, num_iters

        total = left_array + right_array
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
