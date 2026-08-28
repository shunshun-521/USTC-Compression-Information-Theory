"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = 1.0 if sa == 0 else sa
    sb = 1.0 if sb == 0 else sb
    return alpha * sa * sb * min(abs(a), abs(b))


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            left_ele = np.array([left_array[base + j], left_array[base + j + interval]])
            right_ele = np.array([right_array[base + j], right_array[base + j + interval]])
            value[base + j] = _f_min_sum(right_ele[1] + left_ele[1], left_ele[0], alpha)
            value[base + j + interval] = _f_min_sum(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = right_array.size
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            left_ele = np.array([left_array[base + j], left_array[base + j + interval]])
            right_ele = np.array([right_array[base + j], right_array[base + j + interval]])
            value[base + j] = _f_min_sum(right_ele[1] + left_ele[1], right_ele[0], alpha)
            value[base + j + interval] = _f_min_sum(left_ele[0], right_ele[0], alpha) + right_ele[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _hard_decision(self, llr_total):
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_idx] = (llr_total[self.info_idx] < 0).astype(int)
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = _bp_update_left(L[:, stage], R[:, stage - 1], stage, alpha)

            for stage in range(1, n + 1):
                R[:, stage] = _bp_update_right(L[:, stage], R[:, stage - 1], stage, alpha)

            total = L[:, 0] + R[:, 0]
            u_hat = self._hard_decision(total)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = self._hard_decision(total)

        return u_hat, num_iters
