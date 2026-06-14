"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    s1 = np.sign(a)
    s2 = np.sign(b)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return alpha * s1 * s2 * np.minimum(np.abs(a), np.abs(b))


def _bp_update_left(left_col, right_col, stage, alpha):
    """从右向左更新 L 消息（left_matrix）。"""
    N = left_col.size
    interval = 1 << (stage - 1)
    out = left_col.copy()
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            ri = li + interval
            l0, l1 = left_col[li], left_col[ri]
            r0, r1 = right_col[li], right_col[ri]
            out[li] = _ms_f(r1 + l1, l0, alpha)
            out[ri] = _ms_f(l0, r0, alpha) + l1
    return out


def _bp_update_right(left_col, right_col, stage, alpha):
    """从左向右更新 R 消息（right_matrix）。"""
    N = left_col.size
    interval = 1 << (stage - 1)
    out = right_col.copy()
    num = N // (interval * 2)
    for i in range(num):
        for j in range(interval):
            li = 2 * i * interval + j
            ri = li + interval
            l0, l1 = left_col[li], left_col[ri]
            r0, r1 = right_col[li], right_col[ri]
            out[li] = _ms_f(r1 + l1, r0, alpha)
            out[ri] = _ms_f(r0, l0, alpha) + r1
    return out


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_orig = llr_ch.copy()
        n = self.n
        N = self.N
        br = bit_reversal_permutation(N)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                stage = n - i
                L[:, stage - 1] = _bp_update_left(
                    L[:, stage], R[:, stage - 1], stage, self.alpha
                )

            for i in range(n):
                stage = i + 1
                R[:, stage] = _bp_update_right(
                    L[:, stage], R[:, stage - 1], stage, self.alpha
                )

            post = L[:, 0] + R[:, 0]
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if post[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it
            num_iters = it

        post = L[:, 0] + R[:, 0]
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if post[idx] >= 0 else 1

        return u_hat, num_iters
