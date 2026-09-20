"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode, polar_generator_matrix


def _minsum_f(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left_col, right_col, stage, alpha):
    """左向（信道->信源）消息更新"""
    N = len(left_col)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = np.array([left_col[idx0], left_col[idx1]])
            right_ele = np.array([right_col[idx0], right_col[idx1]])
            value[idx0] = _minsum_f(right_ele[1] + left_ele[1], left_ele[0], alpha)
            value[idx1] = _minsum_f(left_ele[0], right_ele[0], alpha) + left_ele[1]
    return value


def _bp_update_right(left_col, right_col, stage, alpha):
    """右向（信源->信道）消息更新"""
    N = len(left_col)
    interval = 2 ** (stage - 1)
    num = N // (interval * 2)
    value = np.zeros(N, dtype=np.float64)
    for i in range(num):
        for j in range(interval):
            idx0 = 2 * i * interval + j
            idx1 = idx0 + interval
            left_ele = np.array([left_col[idx0], left_col[idx1]])
            right_ele = np.array([right_col[idx0], right_col[idx1]])
            value[idx0] = _minsum_f(right_ele[1] + left_ele[1], right_ele[0], alpha)
            value[idx1] = _minsum_f(left_ele[0], right_ele[0], alpha) + right_ele[1]
    return value


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = np.inf

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._G = polar_generator_matrix(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr_ch

        right[:, 0] = 0.0
        right[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(n):
                stage = n - i
                left[:, stage - 1] = _bp_update_left(
                    left[:, stage], right[:, stage - 1], stage, self.alpha
                )

            for i in range(n):
                stage = i + 1
                right[:, stage] = _bp_update_right(
                    left[:, stage], right[:, stage - 1], stage, self.alpha
                )

            llr_post = left[:, 0] + right[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if llr_post[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        llr_post = left[:, 0] + right[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_post[idx] >= 0 else 1

        return u_hat, num_iters
