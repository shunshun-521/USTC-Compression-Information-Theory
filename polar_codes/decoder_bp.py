"""
极化码 BP（置信传播）译码器
基于因子图 min-sum（SMS），含早停机制
"""
import math
import numpy as np
def _f_hf_sms(l1, l2, alpha=0.9375):
    s1 = np.sign(l1) or 1.0
    s2 = np.sign(l2) or 1.0
    return alpha * s1 * s2 * min(abs(l1), abs(l2))


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_hf_sms(right[1] + left[1], left[0], alpha)
    value[1] = _f_hf_sms(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = _f_hf_sms(right[1] + left[1], right[0], alpha)
    value[1] = _f_hf_sms(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, left_array_n, alpha):
    n = left_array_n
    N = left_array.size
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            get_value = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, left_array_n, alpha):
    n = left_array_n
    N = left_array.size
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            right_ele = np.array(
                [right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]]
            )
            get_value = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


class BPDecoder:
    """BP 译码器（自然信道顺序 LLR）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._G = None

    def _generator(self):
        if self._G is None:
            n = self.n
            G = np.array([[1, 0], [1, 1]], dtype=np.int8)
            Gn = np.array([[1]], dtype=np.int8)
            for _ in range(n):
                Gn = np.kron(Gn, G)
            self._G = Gn
        return self._G

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch

        frozen_bit_val = 0
        temp_value = (1 - 2 * frozen_bit_val) * np.inf
        right_init = np.array(
            [temp_value if i not in self.info_idx else 0.0 for i in range(N)], dtype=np.float64
        )
        right_matrix[:, 0] = right_init

        G = self._generator()
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_d_llr < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = (x_d_llr < 0).astype(np.int8)
            x_g = (u_hat @ G) % 2
            if np.array_equal(x_g, x_d):
                num_iters = it
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_d_llr < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
