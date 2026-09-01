"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr


def _f_sms(a, b, alpha=0.9375):
    s1 = 1.0 if a >= 0 else -1.0
    s2 = 1.0 if b >= 0 else -1.0
    return alpha * s1 * s2 * min(abs(a), abs(b))


def _element_update_left(left, right, alpha=0.9375):
    return np.array(
        [
            _f_sms(right[1] + left[1], left[0], alpha),
            _f_sms(left[0], right[0], alpha) + left[1],
        ]
    )


def _element_update_right(left, right, alpha=0.9375):
    return np.array(
        [
            _f_sms(right[1] + left[1], right[0], alpha),
            _f_sms(left[0], right[0], alpha) + right[1],
        ]
    )


def _bp_update_left(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 2 ** (layer - 1)
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


def _bp_update_right(left_array, right_array, layer, alpha):
    N = len(left_array)
    interval = 2 ** (layer - 1)
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

    LARGE = 1e10

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数。返回 u_hat, num_iters。"""
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = np.asarray(llr_ch, dtype=np.float64)
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(n):
                layer = n - i
                L[:, layer - 1] = _bp_update_left(L[:, layer], R[:, layer - 1], layer, alpha)

            for i in range(n):
                layer = i + 1
                R[:, layer] = _bp_update_right(L[:, layer], R[:, layer - 1], layer, alpha)

            total_llr = L[:, 0] + R[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if total_llr[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if total_llr[idx] >= 0 else 1

        return u_hat, num_iters
