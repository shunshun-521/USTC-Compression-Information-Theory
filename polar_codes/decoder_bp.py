"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


def _ms_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self._perm = bit_reversal_permutation(N)

    def _prepare_llr(self, llr_ch):
        return np.asarray(llr_ch, dtype=np.float64)[self._perm]

    def _element_update_left(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = _ms_f(right[1] + left[1], left[0], self.alpha)
        value[1] = _ms_f(left[0], right[0], self.alpha) + left[1]
        return value

    def _element_update_right(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = _ms_f(right[1] + left[1], right[0], self.alpha)
        value[1] = _ms_f(left[0], right[0], self.alpha) + right[1]
        return value

    def _bp_update_left(self, left_array, right_array, layer):
        N = left_array.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]], dtype=np.float64)
                right_ele = np.array([right_array[idx], right_array[idx + interval]], dtype=np.float64)
                out = self._element_update_left(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, layer):
        N = left_array.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]], dtype=np.float64)
                right_ele = np.array([right_array[idx], right_array[idx + interval]], dtype=np.float64)
                out = self._element_update_right(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def decode(self, llr_ch):
        llr_ch = self._prepare_llr(llr_ch)
        n = self.n
        N = self.N

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_bits == 1, 0] = self._large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = self._bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = self._bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if u_llr[i] >= 0 else 1
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = 0 if u_llr[i] >= 0 else 1
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
