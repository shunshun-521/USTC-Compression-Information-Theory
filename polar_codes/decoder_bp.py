"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_ms(a, b, alpha):
    sa = np.sign(a) if np.sign(a) != 0 else 1
    sb = np.sign(b) if np.sign(b) != 0 else 1
    return alpha * sa * sb * min(abs(a), abs(b))


def _element_update_left(left, right, alpha):
    return np.array(
        [
            _f_ms(right[1] + left[1], left[0], alpha),
            _f_ms(left[0], right[0], alpha) + left[1],
        ]
    )


def _element_update_right(left, right, alpha):
    return np.array(
        [
            _f_ms(right[1] + left[1], right[0], alpha),
            _f_ms(left[0], right[0], alpha) + right[1],
        ]
    )


def _bp_update_left(left_array, right_array, stage, alpha):
    N = left_array.size
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
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
    interval = 1 << (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
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
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha
        LARGE = 1e6

        left_arrays = [np.zeros(N)]
        right_arrays = [np.zeros(N)]
        left_arrays.extend([np.zeros(N) for _ in range(n)])
        right_arrays.extend([np.zeros(N) for _ in range(n)])
        left_arrays[n] = llr_ch.copy()
        right_arrays[0][self.frozen_idx] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                left_arrays[stage - 1] = _bp_update_left(
                    left_arrays[stage], right_arrays[stage], stage, alpha
                )
            for stage in range(1, n + 1):
                right_arrays[stage] = _bp_update_right(
                    left_arrays[stage - 1], right_arrays[stage - 1], stage, alpha
                )

            posterior = left_arrays[0] + right_arrays[0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        posterior = left_arrays[0] + right_arrays[0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
