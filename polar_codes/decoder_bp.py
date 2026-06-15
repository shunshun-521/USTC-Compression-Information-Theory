"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _permute_llr_for_decode, _frozen_mask_to_info


def _bp_element_update_left(left, right, alpha):
    def fms(a, b):
        sa, sb = np.sign(a), np.sign(b)
        sa = 1 if sa == 0 else sa
        sb = 1 if sb == 0 else sb
        return alpha * sa * sb * min(abs(a), abs(b))

    out = np.zeros(2)
    out[0] = fms(right[1] + left[1], left[0])
    out[1] = fms(left[0], right[0]) + left[1]
    return out


def _bp_element_update_right(left, right, alpha):
    def fms(a, b):
        sa, sb = np.sign(a), np.sign(b)
        sa = 1 if sa == 0 else sa
        sb = 1 if sb == 0 else sb
        return alpha * sa * sb * min(abs(a), abs(b))

    out = np.zeros(2)
    out[0] = fms(right[1] + left[1], right[0])
    out[1] = fms(left[0], right[0]) + right[1]
    return out


def _bp_update_left(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _bp_element_update_left(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


def _bp_update_right(left_array, right_array, layer_n, alpha):
    N = left_array.size
    interval = 2 ** (layer_n - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx = 2 * i * interval + j
            left_ele = np.array([left_array[idx], left_array[idx + interval]])
            right_ele = np.array([right_array[idx], right_array[idx + interval]])
            out = _bp_element_update_right(left_ele, right_ele, alpha)
            value[idx] = out[0]
            value[idx + interval] = out[1]
    return value


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.information_pos = _frozen_mask_to_info(self.frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = _permute_llr_for_decode(llr_ch)

        L = np.zeros((self.N, self.n + 1))
        R = np.zeros((self.N, self.n + 1))
        L[:, self.n] = llr_perm
        for i in range(self.N):
            R[i, 0] = 0.0 if i in self.information_pos else self.large

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(self.n):
                L[:, self.n - i - 1] = _bp_update_left(
                    L[:, self.n - i], R[:, self.n - i - 1], self.n - i, self.alpha
                )
            for i in range(self.n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            posterior = L[:, 0] + R[:, 0]
            u_hat = np.array([0 if posterior[i] >= 0 else 1 for i in range(self.N)], dtype=int)
            for i in range(self.N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        return u_hat, num_iters
