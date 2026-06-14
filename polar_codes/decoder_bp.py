"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _minsum_f(a, b, alpha):
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = np.where(s1 == 0, 1.0, s1)
    s2 = np.where(s2 == 0, 1.0, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b))


def _element_update_left(left, right, alpha):
    out = np.zeros(2, dtype=np.float64)
    out[0] = _minsum_f(right[1] + left[1], left[0], alpha)
    out[1] = _minsum_f(left[0], right[0], alpha) + left[1]
    return out


def _element_update_right(left, right, alpha):
    out = np.zeros(2, dtype=np.float64)
    out[0] = _minsum_f(right[1] + left[1], right[0], alpha)
    out[1] = _minsum_f(left[0], right[0], alpha) + right[1]
    return out


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            upd = _element_update_left(left_ele, right_ele, alpha)
            value[base] = upd[0]
            value[base + interval] = upd[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            upd = _element_update_right(left_ele, right_ele, alpha)
            value[base] = upd[0]
            value[base + interval] = upd[1]
    return value


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e9

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr_ch[self.rev]

        right[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                right[i, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                stage = n - i
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], stage, self.alpha
                )

            for i in range(n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )

            total = left[:, 0] + right[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            total = left[:, 0] + right[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1
            num_iters = self.max_iter

        return u_hat, num_iters
