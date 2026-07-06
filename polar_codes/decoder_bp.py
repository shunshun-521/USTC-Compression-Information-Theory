"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _gen_matrix(n):
    F = np.array([[1, 0], [1, 1]])
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


def _f_ms(a, b, alpha=0.9375):
    sa = 1 if np.sign(a) == 0 else np.sign(a)
    sb = 1 if np.sign(b) == 0 else np.sign(b)
    return alpha * sa * sb * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    return np.array([
        _f_ms(right[1] + left[1], left[0], alpha),
        _f_ms(left[0], right[0], alpha) + left[1],
    ])


def _element_update_right(left, right, alpha):
    return np.array([
        _f_ms(right[1] + left[1], right[0], alpha),
        _f_ms(left[0], right[0], alpha) + right[1],
    ])


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
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


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = _gen_matrix(self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        inf = 1e9

        left = np.zeros((N, n + 1))
        right = np.zeros((N, n + 1))
        left[:, n] = llr_ch

        for i in range(N):
            if self.frozen_bits[i]:
                right[i, 0] = inf
            else:
                right[i, 0] = 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left[:, n - i - 1] = _bp_update_left(
                    left[:, n - i], right[:, n - i - 1], n - i, self.alpha
                )
            for i in range(n):
                right[:, i + 1] = _bp_update_right(
                    left[:, i + 1], right[:, i], i + 1, self.alpha
                )

            u_llr = left[:, 0] + right[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_llr = left[:, n] + right[:, n]
            x_hat_hard = (x_llr < 0).astype(int)
            x_from_u = (u_hat @ self.G) % 2

            if np.array_equal(x_from_u, x_hat_hard):
                num_iters = it
                break

        u_llr = left[:, 0] + right[:, 0]
        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
