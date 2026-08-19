"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, generator_matrix
from decoder_sc import f_operation


def _f_min_sum(L1, L2, alpha=0.9375):
    """带 alpha 修正的 min-sum f 运算"""
    s1 = 1.0 if L1 == 0 else np.sign(L1)
    s2 = 1.0 if L2 == 0 else np.sign(L2)
    return alpha * s1 * s2 * min(abs(L1), abs(L2))


def _element_update_left(left, right, alpha):
    """BP 左向（L）消息单元更新"""
    value = np.zeros(2)
    value[0] = _f_min_sum(right[1] + left[1], left[0], alpha)
    value[1] = _f_min_sum(left[0], right[0], alpha) + left[1]
    return value


def _element_update_right(left, right, alpha):
    """BP 右向（R）消息单元更新"""
    value = np.zeros(2)
    value[0] = _f_min_sum(right[1] + left[1], right[0], alpha)
    value[1] = _f_min_sum(left[0], right[0], alpha) + right[1]
    return value


def _bp_update_left(left_array, right_array, stage, alpha):
    """BP 左矩阵更新"""
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = np.array([left_array[idx0], left_array[idx1]])
            right_ele = np.array([right_array[idx0], right_array[idx1]])
            out = _element_update_left(left_ele, right_ele, alpha)
            value[idx0] = out[0]
            value[idx1] = out[1]
    return value


def _bp_update_right(left_array, right_array, stage, alpha):
    """BP 右矩阵更新"""
    N = len(left_array)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = np.array([left_array[idx0], left_array[idx1]])
            right_ele = np.array([right_array[idx0], right_array[idx1]])
            out = _element_update_right(left_ele, right_ele, alpha)
            value[idx0] = out[0]
            value[idx1] = out[1]
    return value


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.G = generator_matrix(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                L[:, n - i - 1] = _bp_update_left(
                    L[:, n - i], R[:, n - i - 1], n - i, self.alpha
                )

            for i in range(n):
                R[:, i + 1] = _bp_update_right(
                    L[:, i + 1], R[:, i], i + 1, self.alpha
                )

            u_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if u_llr[i] >= 0 else 1

            x_llr = L[:, n] + R[:, n]
            x_hat_hard = (x_llr < 0).astype(int)
            x_from_u = polar_encode(u_hat)
            if np.array_equal(x_from_u, x_hat_hard):
                num_iters = it
                break

        u_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_llr[i] >= 0 else 1

        return u_hat, num_iters
