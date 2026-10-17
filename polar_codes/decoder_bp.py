"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
"""
import numpy as np
from encoder import polar_encode, polar_generator_matrix
from decoder_sc import f_operation, _align_llr_to_generator


def _f_min_sum(a, b, alpha=0.9375):
    s1 = np.sign(a) if a != 0 else 1
    s2 = np.sign(b) if b != 0 else 1
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    v = np.zeros(2)
    v[0] = _f_min_sum(right[1] + left[1], left[0], alpha)
    v[1] = _f_min_sum(left[0], right[0], alpha) + left[1]
    return v


def _element_update_right(left, right, alpha):
    v = np.zeros(2)
    v[0] = _f_min_sum(right[1] + left[1], right[0], alpha)
    v[1] = _f_min_sum(left[0], right[0], alpha) + right[1]
    return v


def _bp_update_left(left_array, right_array, left_array_n, alpha):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            re = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            get_value = _element_update_left(le, re, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, left_array_n, alpha):
    N = left_array.size
    interval = 2 ** (left_array_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            re = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            get_value = _element_update_right(le, re, alpha)
            value[2 * i * interval + j] = get_value[0]
            value[2 * i * interval + j + interval] = get_value[1]
    return value


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, bit_reverse_codeword=True):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.bit_reverse_codeword = bit_reverse_codeword
        self.G = polar_generator_matrix(N, with_bit_reverse=bit_reverse_codeword)

    def decode(self, llr_ch):
        llr = _align_llr_to_generator(np.asarray(llr_ch, dtype=np.float64), self.bit_reverse_codeword)
        N, n = self.N, self.n
        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = llr
        large = 1e6
        for i in range(N):
            right[i, 0] = 0.0 if not self.frozen_bits[i] else large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(left[:, n - i], right[:, n - i - 1], n - i, self.alpha)
            for i in range(n):
                right[:, i + 1] = _bp_update_right(left[:, i + 1], right[:, i], i + 1, self.alpha)

            post = left[:, 0] + right[:, 0]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat, bit_reverse=self.bit_reverse_codeword)
            x_hard = (llr < 0).astype(int)
            num_iters = it
            if np.array_equal(x_hat, x_hard):
                break

        return u_hat, num_iters
