"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha):
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = 1.0 if s1 == 0 else s1
    s2 = 1.0 if s2 == 0 else s2
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    return np.array([
        _minsum_f(right[1] + left[1], left[0], alpha),
        _minsum_f(left[0], right[0], alpha) + left[1],
    ])


def _element_update_right(left, right, alpha):
    return np.array([
        _minsum_f(right[1] + left[1], right[0], alpha),
        _minsum_f(left[0], right[0], alpha) + right[1],
    ])


def _bp_update_left(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 1 << (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            base = 2 * i * interval + j
            left_ele = np.array([left_array[base], left_array[base + interval]])
            right_ele = np.array([right_array[base], right_array[base + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[base] = out[0]
            value[base + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 1 << (layer - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
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
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N, n = self.N, self.n
        frozen = self.frozen_bits

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[frozen, 0] = np.inf

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                layer = n - i
                L[:, layer - 1] = _bp_update_left(L[:, layer], R[:, layer - 1], layer, self.alpha)

            for i in range(n):
                layer = i + 1
                R[:, layer] = _bp_update_right(L[:, layer], R[:, layer - 1], layer, self.alpha)

            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        posterior = L[:, 0] + R[:, 0]
        u_hat = (posterior < 0).astype(int)
        u_hat[frozen] = 0

        return u_hat, num_iters
