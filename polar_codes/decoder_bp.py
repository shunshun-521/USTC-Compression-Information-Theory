"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, channel_llr_to_decoder


def _f_hf_sms(l1, l2, alpha):
    s1 = np.sign(l1) if l1 != 0 else 1.0
    s2 = np.sign(l2) if l2 != 0 else 1.0
    return alpha * s1 * s2 * min(abs(l1), abs(l2))


def _element_update_left(left, right, alpha):
    return np.array(
        [
            _f_hf_sms(right[1] + left[1], left[0], alpha),
            _f_hf_sms(left[0], right[0], alpha) + left[1],
        ],
        dtype=np.float64,
    )


def _element_update_right(left, right, alpha):
    return np.array(
        [
            _f_hf_sms(right[1] + left[1], right[0], alpha),
            _f_hf_sms(left[0], right[0], alpha) + right[1],
        ],
        dtype=np.float64,
    )


def _bp_update_left(left_array, right_array, layer_n, alpha):
    n = left_array.size
    interval = 2 ** (layer_n - 1)
    num = n // (interval * 2)
    value = np.zeros(n, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    n = left_array.size
    interval = 2 ** (layer_n - 1)
    num = n // (interval * 2)
    value = np.zeros(n, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_internal = channel_llr_to_decoder(llr_ch)
        N = self.N
        n = self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_internal

        temp = np.where(
            self.frozen_bits == 1,
            -np.inf,
            0.0,
        )
        right_matrix[:, 0] = temp

        num_iters = self.max_iter
        hard_ch = (np.asarray(llr_ch) < 0).astype(int)

        for it in range(self.max_iter):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, alpha
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.where(u_llr < 0, 1, 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.where(u_llr < 0, 1, 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
