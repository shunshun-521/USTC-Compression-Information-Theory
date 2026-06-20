"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import build_generator_matrix


class BPDecoder:
    """BP 译码器（因子图 min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits)
        self.frozen_bits = fb.astype(bool) if fb.dtype != bool else fb.copy()
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = build_generator_matrix(N)

    def _f_ms(self, x, y):
        s1 = np.sign(x)
        s2 = np.sign(y)
        s1 = 1 if s1 == 0 else s1
        s2 = 1 if s2 == 0 else s2
        return self.alpha * s1 * s2 * min(abs(x), abs(y))

    def _element_update_left(self, left, right):
        out = np.zeros(2)
        out[0] = self._f_ms(right[1] + left[1], left[0])
        out[1] = self._f_ms(left[0], right[0]) + left[1]
        return out

    def _element_update_right(self, left, right):
        out = np.zeros(2)
        out[0] = self._f_ms(right[1] + left[1], right[0])
        out[1] = self._f_ms(left[0], right[0]) + right[1]
        return out

    def _bp_update_left(self, left_arr, right_arr, layer):
        N = left_arr.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_arr[idx], left_arr[idx + interval]])
                right_ele = np.array([right_arr[idx], right_arr[idx + interval]])
                get_val = self._element_update_left(left_ele, right_ele)
                value[idx] = get_val[0]
                value[idx + interval] = get_val[1]
        return value

    def _bp_update_right(self, left_arr, right_arr, layer):
        N = left_arr.size
        interval = 1 << (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                left_ele = np.array([left_arr[idx], left_arr[idx + interval]])
                right_ele = np.array([right_arr[idx], right_arr[idx + interval]])
                get_val = self._element_update_right(left_ele, right_ele)
                value[idx] = get_val[0]
                value[idx + interval] = get_val[1]
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        inf = 1e10

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch

        frozen_val = 0
        temp_value = (1 - 2 * frozen_val) * inf
        right_matrix[:, 0] = np.array(
            [temp_value if i not in self.info_indices else 0.0 for i in range(N)]
        )

        num_iters = 0
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

            num_iters = it
            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.array([0 if u_llr[i] >= 0 else 1 for i in range(N)])

            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = np.array([0 if x_llr[i] >= 0 else 1 for i in range(N)])
            x_enc = (u_hat @ self.G) % 2

            if np.array_equal(x_enc, x_hard):
                break

        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
