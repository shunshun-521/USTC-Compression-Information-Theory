"""
极化码 BP（置信传播）译码器
因子图 min-sum（SMS）更新，含早停
"""
import math
import numpy as np

from encoder import polar_encode


def _f_sms(a, b, alpha=0.9375):
    s1 = np.sign(a)
    s2 = np.sign(b)
    if s1 == 0:
        s1 = 1.0
    if s2 == 0:
        s2 = 1.0
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    v = np.zeros(2)
    v[0] = _f_sms(right[1] + left[1], left[0], alpha)
    v[1] = _f_sms(left[0], right[0], alpha) + left[1]
    return v


def _element_update_right(left, right, alpha):
    v = np.zeros(2)
    v[0] = _f_sms(right[1] + left[1], right[0], alpha)
    v[1] = _f_sms(left[0], right[0], alpha) + right[1]
    return v


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            re = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            out = _element_update_left(le, re, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = int(N / (interval * 2))
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            le = np.array([left_array[2 * i * interval + j], left_array[2 * i * interval + j + interval]])
            re = np.array([right_array[2 * i * interval + j], right_array[2 * i * interval + j + interval]])
            out = _element_update_right(le, re, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = llr_ch
        inf = set(self.info_indices.tolist())
        right[:, 0] = [0.0 if i in inf else np.inf for i in range(N)]

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(left[:, n - i], right[:, n - i - 1], n - i, self.alpha)
            for i in range(n):
                right[:, i + 1] = _bp_update_right(left[:, i + 1], right[:, i], i + 1, self.alpha)

            total = left[:, 0] + right[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        return u_hat, num_iters
