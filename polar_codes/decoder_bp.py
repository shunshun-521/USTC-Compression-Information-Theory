"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import _preprocess_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _element_update_left(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = self.alpha * np.sign(right[1] + left[1]) * np.sign(left[0]) * min(
            abs(right[1] + left[1]), abs(left[0])
        )
        value[1] = self.alpha * np.sign(left[0]) * np.sign(right[0]) * min(
            abs(left[0]), abs(right[0])
        ) + left[1]
        return value

    def _element_update_right(self, left, right):
        value = np.zeros(2, dtype=np.float64)
        value[0] = self.alpha * np.sign(right[1] + left[1]) * np.sign(right[0]) * min(
            abs(right[1] + left[1]), abs(right[0])
        )
        value[1] = self.alpha * np.sign(left[0]) * np.sign(right[0]) * min(
            abs(left[0]), abs(right[0])
        ) + right[1]
        return value

    def _bp_update_left(self, left_array, right_array, stage):
        N = left_array.size
        interval = 2 ** (stage - 1)
        num = N // (interval * 2)
        value = left_array.copy()
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]])
                right_ele = np.array([right_array[idx], right_array[idx + interval]])
                out = self._element_update_left(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, stage):
        N = left_array.size
        interval = 2 ** (stage - 1)
        num = N // (interval * 2)
        value = right_array.copy()
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_array[idx], left_array[idx + interval]])
                right_ele = np.array([right_array[idx], right_array[idx + interval]])
                out = self._element_update_right(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _preprocess_llr(llr_ch)
        n = self.n
        N = self.N

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_bits, 0] = self.large

        num_iters = 0
        hard_ch = hard_decision_llr(llr_ch)

        for it in range(self.max_iter):
            num_iters = it + 1
            for i in range(n):
                stage = n - i
                left_matrix[:, n - i - 1] = self._bp_update_left(
                    left_matrix[:, stage], right_matrix[:, stage - 1], stage
                )
            for i in range(n):
                stage = i + 1
                right_matrix[:, stage] = self._bp_update_right(
                    left_matrix[:, stage], right_matrix[:, stage - 1], stage
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[u_llr >= 0] = 0
            u_hat[u_llr < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[u_llr >= 0] = 0
        u_hat[u_llr < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
