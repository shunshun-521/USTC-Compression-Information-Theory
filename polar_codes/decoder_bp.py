"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scalar = a.ndim == 0
    if scalar:
        a = np.array([a])
        b = np.array([b])
    s1 = np.sign(a).copy()
    s2 = np.sign(b).copy()
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    result = alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b))
    return result[0] if scalar else result


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            l0, l1 = left_array[idx0], left_array[idx1]
            r0, r1 = right_array[idx0], right_array[idx1]
            value[idx0] = _f_min_sum(r1 + l1, l0, alpha)
            value[idx1] = _f_min_sum(l0, r0, alpha) + l1
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = len(left_array)
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            l0, l1 = left_array[idx0], left_array[idx1]
            r0, r1 = right_array[idx0], right_array[idx1]
            value[idx0] = _f_min_sum(r1 + l1, r0, alpha)
            value[idx1] = _f_min_sum(l0, r0, alpha) + r1
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N = self.N
        n = self.n
        llr_br = llr_ch[self.rev]

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_br
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter

        for iteration in range(self.max_iter):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1],
                    n - i, self.alpha)

            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i],
                    i + 1, self.alpha)

            total_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                num_iters = iteration + 1
                break
        else:
            num_iters = self.max_iter

        total_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1

        return u_hat, num_iters
