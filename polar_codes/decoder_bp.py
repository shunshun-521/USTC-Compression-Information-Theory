"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _info_positions


def _bp_update_left(left_array, right_array, stage):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            a0 = left_array[2 * i * interval + j]
            a1 = left_array[2 * i * interval + j + interval]
            b0 = right_array[2 * i * interval + j]
            f0 = alpha * f_operation(b0 + a1, a0)
            f1 = alpha * f_operation(a0, b0) + a1
            value[2 * i * interval + j] = f0
            value[2 * i * interval + j + interval] = f1
    return value


def _bp_update_right(left_array, right_array, stage):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            a0 = left_array[2 * i * interval + j]
            a1 = left_array[2 * i * interval + j + interval]
            b0 = right_array[2 * i * interval + j]
            b1 = right_array[2 * i * interval + j + interval]
            f0 = alpha * f_operation(b1 + a1, b0)
            f1 = alpha * f_operation(b0, a0) + b1
            value[2 * i * interval + j] = f0
            value[2 * i * interval + j + interval] = f1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = _info_positions(self.frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        y_llr = llr_ch[self._br]

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = y_llr

        large = 1e6
        for i in range(N):
            if self.frozen_bits[i] == 1:
                R[i, 0] = large
            else:
                R[i, 0] = 0.0

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i)
            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1)

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i] == 1:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (y_llr < 0).astype(int)
            x_hard_br = np.zeros(N, dtype=int)
            for j in range(N):
                x_hard_br[self._br[j]] = x_hard[j]
            if np.array_equal(x_hat, x_hard_br):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i] == 1:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1

        return u_hat, num_iters
