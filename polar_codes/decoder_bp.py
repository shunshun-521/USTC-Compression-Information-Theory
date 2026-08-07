"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left, right, layer_n, alpha):
    N = len(left)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            la0 = left[2 * i * interval + j]
            la1 = left[2 * i * interval + j + interval]
            ra0 = right[2 * i * interval + j]
            ra1 = right[2 * i * interval + j + interval]
            value[2 * i * interval + j] = _f_min_sum(ra1 + la1, la0, alpha)
            value[2 * i * interval + j + interval] = _f_min_sum(la0, ra0, alpha) + la1
    return value


def _bp_update_right(left, right, layer_n, alpha):
    N = len(left)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            la0 = left[2 * i * interval + j]
            la1 = left[2 * i * interval + j + interval]
            ra0 = right[2 * i * interval + j]
            ra1 = right[2 * i * interval + j + interval]
            value[2 * i * interval + j] = _f_min_sum(ra1 + la1, ra0, alpha)
            value[2 * i * interval + j + interval] = _f_min_sum(ra0, la1, alpha) + ra1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_orig[rev]

        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i, alpha)
            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, alpha)

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
