"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left, right, stage, alpha):
    N = len(left)
    interval = 1 << (stage - 1)
    num = N // (2 * interval)
    out = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left[base], left[base + interval]
            r0, r1 = right[base], right[base + interval]
            out[base] = _f_min_sum(r1 + l1, l0, alpha)
            out[base + interval] = _f_min_sum(l0, r0, alpha) + l1
    return out


def _bp_update_right(left, right, stage, alpha):
    N = len(left)
    interval = 1 << (stage - 1)
    num = N // (2 * interval)
    out = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            l0, l1 = left[base], left[base + interval]
            r0, r1 = right[base], right[base + interval]
            out[base] = _f_min_sum(r1 + l1, r0, alpha)
            out[base + interval] = _f_min_sum(l0, r0, alpha) + r1
    return out


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
        N, n = self.N, self.n
        alpha = self.alpha

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr_ch
        right[:, 0] = 0.0
        right[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                left[:, stage - 1] = _bp_update_left(
                    left[:, stage], right[:, stage - 1], stage, alpha
                )

            for stage in range(1, n + 1):
                right[:, stage] = _bp_update_right(
                    left[:, stage], right[:, stage - 1], stage, alpha
                )

            num_iters = it
            total = left[:, 0] + right[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = left[:, 0] + right[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
