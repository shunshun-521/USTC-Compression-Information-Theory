"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr


def _boxplus_min_sum(x, y, alpha):
    """min-sum 近似的 box-plus"""
    sa = np.sign(x)
    sb = np.sign(y)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器（参考 Sionna/Arikan 因子图消息传递）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.llr_max = 19.3

    def _bp_iteration(self, llr_ch, msg_l_prev, msg_r_in):
        """单次 BP 迭代（左→右再右→左）"""
        n = self.n_stages
        N = self.N
        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for ind_s in range(n):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s

            if ind_s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif msg_l_prev is None:
                l1_in = np.zeros(N // 2)
                l2_in = np.zeros(N // 2)
            else:
                l_in = msg_l_prev[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = _boxplus_min_sum(r1_in, l2_in + r2_in, self.alpha)
            r2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + r2_in

            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

        for ind_s in range(n - 1, -1, -1):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

            if ind_s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = _boxplus_min_sum(l1_in, l2_in + r2_in, self.alpha)
            l2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + l2_in

            msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

        return msg_l

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.clip(
            np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max
        )
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            msg_l = self._bp_iteration(llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l

            llr_out = msg_l[0]
            for i in range(N):
                u_hat[i] = 0 if llr_out[i] >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, hard):
                num_iters = it + 1
                break

        llr_out = msg_l_prev[0]
        for i in range(N):
            u_hat[i] = 0 if llr_out[i] >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
