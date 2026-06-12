"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    s1 = np.sign(x)
    s2 = np.sign(y)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(x), np.abs(y))


def _bp_update_left(left_col, right_col, layer, alpha):
    """从右向左更新 L 消息。"""
    N = len(left_col)
    interval = 2 ** (layer - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (2 * interval)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_col[idx], left_col[idx + interval]])
            right_ele = np.array([right_col[idx], right_col[idx + interval]])
            out[idx] = _f_min_sum(right_ele[1] + left_ele[1], left_ele[0], alpha)
            out[idx + interval] = _f_min_sum(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return out


def _bp_update_right(left_col, right_col, layer, alpha):
    """从左向右更新 R 消息。"""
    N = len(left_col)
    interval = 2 ** (layer - 1)
    out = np.zeros(N, dtype=np.float64)
    num = N // (2 * interval)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_col[idx], left_col[idx + interval]])
            right_ele = np.array([right_col[idx], right_col[idx + interval]])
            out[idx] = _f_min_sum(right_ele[1] + left_ele[1], right_ele[0], alpha)
            out[idx + interval] = _f_min_sum(left_ele[0], right_ele[0], alpha) + right_ele[1]
    return out


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

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(
                    L[:, n - i], R[:, n - i - 1], n - i, self.alpha
                )

            for i in range(n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
