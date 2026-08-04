"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        sa, sb = np.sign(a), np.sign(b)
        sa = np.where(sa == 0, 1, sa)
        sb = np.where(sb == 0, 1, sb)
        return self.alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def _element_update_left(self, left, right):
        out = np.zeros(2)
        out[0] = self._f_min_sum(right[1] + left[1], left[0])
        out[1] = self._f_min_sum(left[0], right[0]) + left[1]
        return out

    def _element_update_right(self, left, right):
        out = np.zeros(2)
        out[0] = self._f_min_sum(right[1] + left[1], right[0])
        out[1] = self._f_min_sum(left[0], right[0]) + right[1]
        return out

    def _bp_update_left(self, left_arr, right_arr, layer):
        N = left_arr.size
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_arr[2 * i * interval + j], left_arr[2 * i * interval + j + interval]])
                right_ele = np.array([right_arr[2 * i * interval + j], right_arr[2 * i * interval + j + interval]])
                get_val = self._element_update_left(left_ele, right_ele)
                value[2 * i * interval + j] = get_val[0]
                value[2 * i * interval + j + interval] = get_val[1]
        return value

    def _bp_update_right(self, left_arr, right_arr, layer):
        N = left_arr.size
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                left_ele = np.array([left_arr[2 * i * interval + j], left_arr[2 * i * interval + j + interval]])
                right_ele = np.array([right_arr[2 * i * interval + j], right_arr[2 * i * interval + j + interval]])
                get_val = self._element_update_right(left_ele, right_ele)
                value[2 * i * interval + j] = get_val[0]
                value[2 * i * interval + j + interval] = get_val[1]
        return value

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev]
        N, n = self.N, self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_idx, 0] = np.inf

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                left_matrix[:, layer - 1] = self._bp_update_left(
                    left_matrix[:, layer], right_matrix[:, layer - 1], layer
                )

            for layer in range(1, n + 1):
                right_matrix[:, layer] = self._bp_update_right(
                    left_matrix[:, layer], right_matrix[:, layer - 1], layer
                )

            posterior = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or posterior[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        posterior = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] or posterior[i] >= 0 else 1

        return u_hat, num_iters
