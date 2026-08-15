"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(x, y, llr_max=19.3):
    x_in = np.clip(x, -llr_max, llr_max)
    y_in = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x_in + y_in)) - np.log(np.exp(x_in) + np.exp(y_in))


def _min_sum(x, y, alpha=0.9375):
    sx = np.sign(x)
    sy = np.sign(y)
    sx = np.where(sx == 0, 1.0, sx)
    sy = np.where(sy == 0, 1.0, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    输入 llr_ch 为信道 LLR（正倾向比特 0）；内部做比特倒序以匹配因子图。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_min_sum = use_min_sum
        self.llr_max = 19.3
        self._br = bit_reversal_permutation(N)

    def _f(self, x, y):
        if self.use_min_sum:
            return _min_sum(x, y, self.alpha)
        return _boxplus(x, y, self.llr_max)

    def _one_iteration(
        self, llr_internal, ind_it, msg_l_prev, msg_r_row, msg_r_in
    ):
        """执行单次 BP 迭代，返回该次迭代最左列 L 消息 soft。"""
        N = self.N
        n_stages = self.n

        for ind_s in range(n_stages):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

            if ind_s == n_stages - 1:
                l1_in = llr_internal[ind_1]
                l2_in = llr_internal[ind_2]
            elif ind_it == 0:
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
                r_in = msg_r_row[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = self._f(r1_in, l2_in + r2_in)
            r2_out = self._f(r1_in, l1_in) + r2_in
            msg_r_row[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

        msg_l_row = [None] * (n_stages + 1)
        for ind_s in range(n_stages - 1, -1, -1):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

            if ind_s == n_stages - 1:
                l1_in = llr_internal[ind_1]
                l2_in = llr_internal[ind_2]
            else:
                l_in = msg_l_row[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r_row[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = self._f(l1_in, l2_in + r2_in)
            l2_out = self._f(r1_in, l1_in) + l2_in
            msg_l_row[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

        return msg_l_row[0], msg_l_row

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self._br]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l_prev = None
        soft = np.zeros(self.N, dtype=np.float64)
        num_iters = self.max_iter

        for ind_it in range(self.max_iter):
            msg_r_row = [None] * (self.n + 1)
            soft, msg_l_prev = self._one_iteration(
                llr_internal, ind_it, msg_l_prev, msg_r_row, msg_r_in
            )

            u_hat = np.zeros(self.N, dtype=int)
            for idx in range(self.N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if soft[idx] > 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        u_hat = np.zeros(self.N, dtype=int)
        for idx in range(self.N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if soft[idx] > 0 else 1

        return u_hat, num_iters
