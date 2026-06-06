"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _element_update_left(self, left, right):
        return np.array(
            [
                self._f_ms(right[1] + left[1], left[0]),
                self._f_ms(left[0], right[0]) + left[1],
            ]
        )

    def _element_update_right(self, left, right):
        return np.array(
            [
                self._f_ms(right[1] + left[1], right[0]),
                self._f_ms(left[0], right[0]) + right[1],
            ]
        )

    def _bp_update_left(self, left_array, right_array, stage):
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
                right_ele = np.array(
                    [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
                )
                out = self._element_update_left(left_ele, right_ele)
                value[2 * i * interval + j] = out[0]
                value[2 * i * interval + j + interval] = out[1]
        return value

    def _bp_update_right(self, left_array, right_array, stage):
        N = self.N
        interval = 1 << (stage - 1)
        num = N // (interval * 2)
        value = np.zeros(N, dtype=np.float64)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
                right_ele = np.array(
                    [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
                )
                out = self._element_update_right(left_ele, right_ele)
                value[2 * i * interval + j] = out[0]
                value[2 * i * interval + j + interval] = out[1]
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = self._bp_update_left(L[:, stage], R[:, stage - 1], stage)

            for stage in range(1, n + 1):
                R[:, stage] = self._bp_update_right(L[:, stage], R[:, stage - 1], stage)

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
