"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _bp_update_left(left_col, right_col, layer_idx):
    """从右到左更新 L 消息（min-sum）。"""
    N = len(left_col)
    n = layer_idx
    interval = 2 ** (n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            left_ele = np.array([left_col[li], left_col[li + interval]])
            right_ele = np.array([right_col[li], right_col[li + interval]])
            v0 = alpha * f_operation(right_ele[1] + left_ele[1], left_ele[0])
            v1 = alpha * f_operation(left_ele[0], right_ele[0]) + left_ele[1]
            value[li] = v0
            value[li + interval] = v1
    return value


def _bp_update_right(left_col, right_col, layer_idx):
    """从左到右更新 R 消息（min-sum）。"""
    N = len(left_col)
    interval = 2 ** (layer_idx - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            left_ele = np.array([left_col[li], left_col[li + interval]])
            right_ele = np.array([right_col[li], right_col[li + interval]])
            v0 = alpha * f_operation(right_ele[1] + left_ele[1], right_ele[0])
            v1 = alpha * f_operation(left_ele[0], right_ele[0]) + right_ele[1]
            value[li] = v0
            value[li + interval] = v1
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(L[:, n - i], R[:, n - i - 1], n - i)

            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1)

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
