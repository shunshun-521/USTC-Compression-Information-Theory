"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation


def _polar_encode_nobr(u):
    x = np.asarray(u, dtype=int).copy()
    N = len(x)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                x[i + j] ^= x[i + j + step]
    return x


def _f_min_sum(a, b, alpha=0.9375):
    return alpha * f_operation(a, b)


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            l0, l1 = left_array[idx], left_array[idx + interval]
            r0, r1 = right_array[idx], right_array[idx + interval]
            value[idx] = _f_min_sum(r1 + l1, l0, alpha)
            value[idx + interval] = _f_min_sum(l0, r0, alpha) + l1
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            l0, l1 = left_array[idx], left_array[idx + interval]
            r0, r1 = right_array[idx], right_array[idx + interval]
            value[idx] = _f_min_sum(r1 + l1, r0, alpha)
            value[idx + interval] = _f_min_sum(l0, r0, alpha) + r1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_pos = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        y_llr = llr_ch[br]

        N, n = self.N, self.n
        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = y_llr
        right_matrix[:, 0] = np.where(
            self.frozen_bits, np.inf, 0.0
        )

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, self.alpha
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            x_reenc = _polar_encode_nobr(u_hat)
            if np.array_equal(x_reenc, x_hard):
                num_iters = it
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
