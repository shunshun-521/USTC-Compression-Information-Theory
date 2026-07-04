"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_minsum(x, y, alpha):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = np.sign(x)
    sy = np.sign(y)
    sx = np.where(sx == 0, 1, sx)
    sy = np.where(sy == 0, 1, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            value[base] = _f_minsum(right_ele[1] + left_ele[1], left_ele[0], alpha)
            value[base + interval] = _f_minsum(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            value[base] = _f_minsum(right_ele[1] + left_ele[1], right_ele[0], alpha)
            value[base + interval] = _f_minsum(right_ele[0], left_ele[1], alpha) + right_ele[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_tree = llr_ch[self.br]
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_tree
        frozen_tree = self.frozen_bits[self.br]
        R[:, 0] = 0.0
        R[frozen_tree, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(
                    L[:, n - i], R[:, n - i - 1], n - i, alpha
                )
            for i in range(n):
                R[:, i + 1] = _bp_update_right(L[:, i + 1], R[:, i], i + 1, alpha)

            total = L[:, 0] + R[:, 0]
            u_tree = (total < 0).astype(int)
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.br] = u_tree
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_tree = (total < 0).astype(int)
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.br] = u_tree
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
