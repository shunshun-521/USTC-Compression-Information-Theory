"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


def _bp_update_left(left_array, right_array, stage):
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left_array[base], left_array[base + interval]
            r0, r1 = right_array[base], right_array[base + interval]
            value[base] = alpha * f_operation(r1 + l1, l0)
            value[base + interval] = alpha * f_operation(l0, r0) + l1
    return value


def _bp_update_right(left_array, right_array, stage):
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left_array[base], left_array[base + interval]
            r0, r1 = right_array[base], right_array[base + interval]
            value[base] = alpha * f_operation(r1 + l1, r0)
            value[base + interval] = alpha * f_operation(r0, l0) + r1
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i)

            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1)

            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
