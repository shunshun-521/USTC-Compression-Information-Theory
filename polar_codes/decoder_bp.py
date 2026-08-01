"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _bp_update_left(left_array, right_array, layer_n):
    """从右向左更新 L 消息"""
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            left_ele = np.array([left_array[li], left_array[li + interval]])
            right_ele = np.array([right_array[li], right_array[li + interval]])
            f0 = alpha * f_operation(right_ele[1] + left_ele[1], left_ele[0])
            f1 = alpha * f_operation(left_ele[0], right_ele[0]) + left_ele[1]
            value[li] = f0
            value[li + interval] = f1
    return value


def _bp_update_right(left_array, right_array, layer_n):
    """从左向右更新 R 消息"""
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    alpha = 0.9375
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            left_ele = np.array([left_array[li], left_array[li + interval]])
            right_ele = np.array([right_array[li], right_array[li + interval]])
            f0 = alpha * f_operation(right_ele[1] + left_ele[1], right_ele[0])
            f1 = alpha * f_operation(left_ele[0], right_ele[0]) + right_ele[1]
            value[li] = f0
            value[li + interval] = f1
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = np.inf
            else:
                R[i, 0] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for layer in range(n):
                L[:, n - layer - 1] = _bp_update_left(L[:, n - layer], R[:, n - layer - 1], n - layer)

            for layer in range(n):
                R[:, layer + 1] = _bp_update_right(L[:, layer + 1], R[:, layer], layer + 1)

            u_llr = L[:, 0] + R[:, 0]
            for i in range(N):
                u_hat[i] = 0 if u_llr[i] >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_llr = L[:, n] + R[:, n]
            x_hard = np.array([0 if x_llr[i] >= 0 else 1 for i in range(N)], dtype=int)
            if np.array_equal(x_hat, x_hard):
                break

        u_llr = L[:, 0] + R[:, 0]
        for i in range(N):
            u_hat[i] = 0 if u_llr[i] >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
