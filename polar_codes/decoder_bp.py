"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _element_update_left(self, left, right):
        value = np.zeros(2)
        value[0] = self._f_ms(right[1] + left[1], left[0])
        value[1] = self._f_ms(left[0], right[0]) + left[1]
        return value

    def _element_update_right(self, left, right):
        value = np.zeros(2)
        value[0] = self._f_ms(right[1] + left[1], right[0])
        value[1] = self._f_ms(left[0], right[0]) + right[1]
        return value

    def _bp_update_left(self, left_array, right_array, layer_n):
        N = len(left_array)
        interval = 2 ** (layer_n - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
                right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
                out = self._element_update_left(left_ele, right_ele)
                value[2 * i * interval + j] = out[0]
                value[2 * i * interval + j + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer_n):
        N = len(left_array)
        interval = 2 ** (layer_n - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
                right_ele = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
                out = self._element_update_right(left_ele, right_ele)
                value[2 * i * interval + j] = out[0]
                value[2 * i * interval + j + interval] = out[1]
        return value

    def decode(self, llr_ch):
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_bits, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = self._bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = self._bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if total[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = left_matrix[:, 0] + right_matrix[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if total[idx] >= 0 else 1

        return u_hat, num_iters
