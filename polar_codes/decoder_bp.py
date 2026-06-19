"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


def _bp_element_update_left(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], left[0])
    value[1] = alpha * f_operation(left[0], right[0]) + left[1]
    return value


def _bp_element_update_right(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = alpha * f_operation(right[1] + left[1], right[0])
    value[1] = alpha * f_operation(left[0], right[0]) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            get_value = _bp_element_update_left(left_ele, right_ele, alpha)
            value[base] = get_value[0]
            value[base + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            get_value = _bp_element_update_right(left_ele, right_ele, alpha)
            value[base] = get_value[0]
            value[base + interval] = get_value[1]
    return value


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.is_info = self.frozen_bits == 0
        self.info_indices = np.where(self.is_info)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        for i in range(N):
            if self.is_info[i]:
                R[i, 0] = 0.0
            else:
                R[i, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(
                    L[:, n - i], R[:, n - i - 1], n - i, self.alpha
                )

            for i in range(n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            total_llr = L[:, 0] + R[:, 0]
            for i in range(N):
                if not self.is_info[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total_llr = L[:, 0] + R[:, 0]
        for i in range(N):
            if not self.is_info[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1

        return u_hat, num_iters
