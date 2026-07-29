"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _bit_reversed


def _f_min_sum(l1, l2, alpha):
    s1 = 1.0 if l1 >= 0 else -1.0
    s2 = 1.0 if l2 >= 0 else -1.0
    return alpha * s1 * s2 * min(abs(l1), abs(l2))


def _element_update_left(left, right, alpha):
    """PE 左向（L）消息更新。"""
    v0 = _f_min_sum(right[1] + left[1], left[0], alpha)
    v1 = _f_min_sum(left[0], right[0], alpha) + left[1]
    return np.array([v0, v1])


def _element_update_right(left, right, alpha):
    """PE 右向（R）消息更新。"""
    v0 = _f_min_sum(right[1] + left[1], right[0], alpha)
    v1 = _f_min_sum(left[0], right[0], alpha) + right[1]
    return np.array([v0, v1])


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._inv_rev = np.argsort(
            [_bit_reversed(i, self.n) for i in range(N)]
        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha
        LARGE = 1e8

        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = llr_ch[self._inv_rev]

        for i in range(N):
            if self.frozen_bits[i]:
                right[i, 0] = LARGE
            else:
                right[i, 0] = 0.0

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for i in range(n):
                stage = n - i
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], stage, alpha
                )
            for i in range(n):
                stage = i + 1
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], stage, alpha
                )

            u_llr = left[:, 0] + right[:, 0]
            u_hat = np.where(u_llr >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_llr = left[:, 0] + right[:, 0]
        u_hat = np.where(u_llr >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
