"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import bit_reversed
from encoder import polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    s1 = 1 if a == 0 else np.sign(a)
    s2 = 1 if b == 0 else np.sign(b)
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _bp_update_left(left_col, right_col, layer_n, alpha):
    N = len(left_col)
    interval = 1 << (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = [left_col[idx0], left_col[idx1]]
            right_ele = [right_col[idx0], right_col[idx1]]
            value[idx0] = _f_min_sum(right_ele[1] + left_ele[1], left_ele[0], alpha)
            value[idx1] = _f_min_sum(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return value


def _bp_update_right(left_col, right_col, layer_n, alpha):
    N = len(left_col)
    interval = 1 << (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = [left_col[idx0], left_col[idx1]]
            right_ele = [right_col[idx0], right_col[idx1]]
            value[idx0] = _f_min_sum(right_ele[1] + left_ele[1], right_ele[0], alpha)
            value[idx1] = _f_min_sum(left_ele[0], right_ele[0], alpha) + right_ele[1]
    return value


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._rev = np.array([bit_reversed(i, self.n) for i in range(N)], dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self._rev]

        N = self.N
        n = self.n
        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr

        for i in range(N):
            if self.frozen_bits[i] == 0:
                right[i, 0] = 0.0
            else:
                right[i, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )

            llr_dec = left[:, 0] + right[:, 0]
            for i in range(N):
                u_hat[i] = 0 if llr_dec[i] >= 0 else 1
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        llr_dec = left[:, 0] + right[:, 0]
        for i in range(N):
            u_hat[i] = 0 if llr_dec[i] >= 0 else 1
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
