"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left_array, right_array, stage, alpha):
    N = len(left_array)
    interval = 1 << (stage - 1)
    value = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0 = left_array[base]
            l1 = left_array[base + interval]
            r0 = right_array[base]
            r1 = right_array[base + interval]
            value[base] = _minsum_f(r1 + l1, l0, alpha)
            value[base + interval] = _minsum_f(l0, r0, alpha) + l1
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = len(left_array)
    interval = 1 << (stage - 1)
    value = np.zeros(N, dtype=np.float64)
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0 = left_array[base]
            l1 = left_array[base + interval]
            r0 = right_array[base]
            r1 = right_array[base + interval]
            value[base] = _minsum_f(r1 + l1, r0, alpha)
            value[base + interval] = _minsum_f(l0, r0, alpha) + r1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = _bp_update_left(L[:, stage], R[:, stage - 1], stage, self.alpha)
            for stage in range(1, n + 1):
                R[:, stage] = _bp_update_right(L[:, stage], R[:, stage - 1], stage, self.alpha)

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
