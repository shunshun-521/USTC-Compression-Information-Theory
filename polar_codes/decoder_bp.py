"""
极化码 BP（置信传播）译码器
基于因子图（Sionna/Arikan 索引），min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _boxplus(x, y, llr_max=19.3):
    """精确 LLR 合并（boxplus）"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.setdiff1d(np.arange(N), self.frozen_pos)
        self.max_iter = max_iter
        self.alpha = alpha
        self._brp = bit_reversal_permutation(N)

    def _combine(self, x, y):
        if self.alpha >= 1.0:
            return _boxplus(x, y, self.LARGE)
        return _f_min_sum(x, y, self.alpha)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.n
        llr = llr_ch[self._brp]

        msg_l = [[None] * (m + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (m + 1) for _ in range(self.max_iter)]
        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            # 左向传播 R（Sionna 索引）
            for ind_s in range(m):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == m - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._combine(r1_in, l2_in + r2_in)
                r2_out = self._combine(r1_in, l1_in) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            # 右向传播 L
            for ind_s in range(m - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == m - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = self._combine(l1_in, l2_in + r2_in)
                l2_out = self._combine(r1_in, l1_in) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            num_iters = ind_it + 1
            u_soft = msg_l[ind_it][0][self.info_pos]
            u_hat[self.info_pos] = np.where(u_soft > 0, 0, 1)
            u_hat[self.frozen_pos] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        if num_iters == 0:
            u_soft = msg_l[self.max_iter - 1][0][self.info_pos]
            u_hat[self.info_pos] = np.where(u_soft > 0, 0, 1)
            u_hat[self.frozen_pos] = 0

        return u_hat, num_iters
