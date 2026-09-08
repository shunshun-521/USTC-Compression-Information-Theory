"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _frozen_to_info, _prepare_llr, f_operation
from encoder import bit_reversal_permutation, polar_encode


LARGE = 1e6


def _f_minsum(a, b, alpha=0.9375):
    return alpha * f_operation(a, b)


def _element_update_left(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = _f_minsum(right[1] + left[1], left[0], alpha)
    value[1] = _f_minsum(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    value = np.zeros(2, dtype=np.float64)
    value[0] = _f_minsum(right[1] + left[1], right[0], alpha)
    value[1] = _f_minsum(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = _frozen_to_info(self.frozen_bits)
        self.info_set = set(int(i) for i in self.info_positions)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr = _prepare_llr(llr_ch, N)

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr

        frozen_val = 0
        right_matrix[:, 0] = np.array(
            [
                (1 - 2 * frozen_val) * LARGE if i not in self.info_set else 0.0
                for i in range(N)
            ],
            dtype=np.float64,
        )

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                    self.alpha,
                )
            for i in range(n):
                right_matrix[:, i + 1] = _bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                    self.alpha,
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = (
                    frozen_val
                    if i not in self.info_set
                    else (0 if u_llr[i] >= 0 else 1)
                )

            x_hat_nat = polar_encode(u_hat)
            x_hat = x_hat_nat[self.rev]
            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            u_hat[i] = (
                frozen_val
                if i not in self.info_set
                else (0 if u_llr[i] >= 0 else 1)
            )

        return u_hat.astype(int), num_iters
