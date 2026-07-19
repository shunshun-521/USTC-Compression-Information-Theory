"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus(x, y):
    """Log-domain box-plus operation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t = np.tanh(x / 2.0) * np.tanh(y / 2.0)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


def _minsum_boxplus(x, y, alpha=0.9375):
    sign = np.sign(x) * np.sign(y)
    mag = np.minimum(np.abs(x), np.abs(y))
    return alpha * sign * mag


class BPDecoder:
    """
    BP 译码器（参考 Sionna/Arikan 因子图消息传递结构）。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._use_minsum = alpha < 1.0

    def _bp(self, a, b):
        if self._use_minsum:
            return _minsum_boxplus(a, b, self.alpha)
        return _boxplus(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LARGE

        num_iters = self.max_iter

        for ind_it in range(self.max_iter):
            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
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

                r1_out = self._bp(r1_in, l2_in + r2_in)
                r2_out = self._bp(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

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

                l1_out = self._bp(l1_in, l2_in + r2_in)
                l2_out = self._bp(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            u_hat = np.zeros(N, dtype=int)
            llr_info = msg_l[0][self.info_pos]
            u_hat[self.info_pos] = (llr_info < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        u_hat = np.zeros(N, dtype=int)
        llr_info = msg_l[0][self.info_pos]
        u_hat[self.info_pos] = (llr_info < 0).astype(int)
        for idx in self.frozen_pos:
            u_hat[idx] = 0

        return u_hat, num_iters
