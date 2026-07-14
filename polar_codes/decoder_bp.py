"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _minsum(a, b, alpha=0.9375):
    s1 = 1 if np.sign(a) == 0 else np.sign(a)
    s2 = 1 if np.sign(b) == 0 else np.sign(b)
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha=0.9375):
    value = np.zeros(2, dtype=np.float64)
    value[0] = _minsum(right[1] + left[1], left[0], alpha)
    value[1] = _minsum(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha=0.9375):
    value = np.zeros(2, dtype=np.float64)
    value[0] = _minsum(right[1] + left[1], right[0], alpha)
    value[1] = _minsum(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, layer_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([
                left_array[2 * i * interval + j],
                left_array[2 * i * interval + j + interval],
            ])
            right_ele = np.array([
                right_array[2 * i * interval + j],
                right_array[2 * i * interval + j + interval],
            ])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha=0.9375):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            left_ele = np.array([
                left_array[2 * i * interval + j],
                left_array[2 * i * interval + j + interval],
            ])
            right_ele = np.array([
                right_array[2 * i * interval + j],
                right_array[2 * i * interval + j + interval],
            ])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[2 * i * interval + j] = out[0]
            value[2 * i * interval + j + interval] = out[1]
    return value


class BPDecoder:
    """
    BP 译码器。
  因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = np.inf
        self.inv_br = np.argsort(bit_reversal_permutation(N))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch[self.inv_br]

        for i in range(N):
            if self.frozen_bits[i]:
                right_matrix[i, 0] = self.large
            else:
                right_matrix[i, 0] = 0.0

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i, alpha
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1, alpha
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if u_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

        return u_hat, num_iters
