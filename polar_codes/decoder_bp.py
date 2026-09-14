"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


def _bp_update_left(left_array, right_array, stage_n, alpha):
    N = left_array.size
    interval = 2 ** (stage_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            idx0 = base + j
            idx1 = base + j + interval
            left_ele = np.array([left_array[idx0], left_array[idx1]])
            right_ele = np.array([right_array[idx0], right_array[idx1]])
            value[idx0] = alpha * f_operation(right_ele[1] + left_ele[1], left_ele[0])
            value[idx1] = alpha * f_operation(left_ele[0], right_ele[0]) + left_ele[1]
    return value


def _bp_update_right(left_array, right_array, stage_n, alpha):
    N = left_array.size
    interval = 2 ** (stage_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for block in range(num):
        base = 2 * block * interval
        for j in range(interval):
            idx0 = base + j
            idx1 = base + j + interval
            left_ele = np.array([left_array[idx0], left_array[idx1]])
            right_ele = np.array([right_array[idx0], right_array[idx1]])
            value[idx0] = alpha * f_operation(right_ele[1] + left_ele[1], right_ele[0])
            value[idx1] = alpha * f_operation(left_ele[0], right_ele[0]) + right_ele[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e10
        br = bit_reversal_permutation(N)
        self.inv_br = np.zeros(N, dtype=int)
        self.inv_br[br] = np.arange(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)

        left[:, n] = llr_ch[self.inv_br]
        right[:, 0] = np.where(self.frozen_bits, -self.large, 0.0)

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

            total = left[:, 0] + right[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = left[:, 0] + right[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
