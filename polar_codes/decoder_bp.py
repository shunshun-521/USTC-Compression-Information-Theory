"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def min_sum_f(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算，带修正因子 alpha"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _element_update_left(left, right, alpha):
    value = np.zeros(2)
    value[0] = min_sum_f(right[1] + left[1], left[0], alpha)
    value[1] = min_sum_f(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2)
    value[0] = min_sum_f(right[1] + left[1], right[0], alpha)
    value[1] = min_sum_f(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, stage, alpha):
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            get_value = _element_update_left(left_ele, right_ele, alpha)
            value[idx] = get_value[0]
            value[idx + interval] = get_value[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            get_value = _element_update_right(left_ele, right_ele, alpha)
            value[idx] = get_value[0]
            value[idx + interval] = get_value[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = np.asarray(llr_ch, dtype=np.float64)
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for iteration in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = _bp_update_left(L[:, stage], R[:, stage], stage, alpha)

            for stage in range(1, n + 1):
                R[:, stage] = _bp_update_right(R[:, stage - 1], L[:, stage], stage, alpha)

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = iteration
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat
