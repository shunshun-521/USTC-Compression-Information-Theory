"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _element_update_left(self, left, right):
        return np.array([
            self._f_ms(right[1] + left[1], left[0]),
            self._f_ms(left[0], right[0]) + left[1],
        ])

    def _element_update_right(self, left, right):
        return np.array([
            self._f_ms(right[1] + left[1], right[0]),
            self._f_ms(left[0], right[0]) + right[1],
        ])

    def _bp_update_left(self, left_array, right_array, layer):
        N = left_array.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_array[base], left_array[base + interval]])
                right_ele = np.array([right_array[base], right_array[base + interval]])
                out = self._element_update_left(left_ele, right_ele)
                value[base] = out[0]
                value[base + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer):
        N = left_array.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                left_ele = np.array([left_array[base], left_array[base + interval]])
                right_ele = np.array([right_array[base], right_array[base + interval]])
                out = self._element_update_right(left_ele, right_ele)
                value[base] = out[0]
                value[base + interval] = out[1]
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        brp = bit_reversal_permutation(N)
        y_llr = llr_ch[brp]

        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = y_llr
        right[:, 0] = 0.0
        right[self.frozen_idx, 0] = np.inf

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left[:, n - i - 1] = self._bp_update_left(left[:, n - i], right[:, n - i - 1], n - i)
            for i in range(n):
                right[:, i + 1] = self._bp_update_right(left[:, i + 1], right[:, i], i + 1)

            total = left[:, 0] + right[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = left[:, 0] + right[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
