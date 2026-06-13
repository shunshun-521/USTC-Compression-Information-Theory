"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from sc_core import f_min_sum_alpha, _frozen_to_info_set


def _bp_update_left(left_array, right_array, layer_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            la0, la1 = left_array[idx0], left_array[idx1]
            ra0, ra1 = right_array[idx0], right_array[idx1]
            value[idx0] = f_min_sum_alpha(ra1 + la1, la0, alpha)
            value[idx1] = f_min_sum_alpha(la0, ra0, alpha) + la1
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            la0, la1 = left_array[idx0], left_array[idx1]
            ra0, ra1 = right_array[idx0], right_array[idx1]
            value[idx0] = f_min_sum_alpha(ra1 + la1, ra0, alpha)
            value[idx1] = f_min_sum_alpha(la0, ra0, alpha) + ra1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_set = set(_frozen_to_info_set(frozen_bits))
        self.max_iter = max_iter
        self.alpha = alpha
        self.n = int(np.log2(N))
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = llr_ch[self.br]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = y_llr
        for i in range(N):
            R[i, 0] = 0.0 if i in self.info_set else self._large

        num_iters = self.max_iter
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

            post = L[:, 0] + R[:, 0]
            for i in range(N):
                if i not in self.info_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if post[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        post = L[:, 0] + R[:, 0]
        for i in range(N):
            if i not in self.info_set:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if post[i] >= 0 else 1

        return u_hat.astype(int), num_iters
