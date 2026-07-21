"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _boxplus_minsum(x, y, alpha=0.9375):
    return alpha * f_operation(x, y)


def _boxplus_exact(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.llr_max = 19.3
        self._boxplus = lambda x, y: _boxplus_minsum(x, y, self.alpha)

    def _run_iteration(self, llr_ch, msg_l_prev, msg_r_prev, msg_r_in, first_iter):
        n_stages = self.n
        N = self.N
        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        for ind_s in range(n_stages):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s

            if ind_s == n_stages - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif first_iter:
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

            r1_out = self._boxplus(r1_in, l2_in + r2_in)
            r2_out = self._boxplus(r1_in, l1_in) + r2_in
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

        for ind_s in range(n_stages - 1, -1, -1):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

            if ind_s == n_stages - 1:
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

            l1_out = self._boxplus(l1_in, l2_in + r2_in)
            l2_out = self._boxplus(r1_in, l1_in) + l2_in
            msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

        return msg_l, msg_r

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        msg_r_prev = None

        for it in range(self.max_iter):
            msg_l, msg_r = self._run_iteration(
                llr_ch, msg_l_prev, msg_r_prev, msg_r_in, first_iter=(it == 0)
            )
            msg_l_prev = msg_l
            msg_r_prev = msg_r

            llr_total = msg_l[0] + msg_r_in
            u_hat = np.zeros(self.N, dtype=int)
            for i in range(self.N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if llr_total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it + 1

        llr_total = msg_l_prev[0] + msg_r_in
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_total[i] >= 0 else 1
        return u_hat, self.max_iter
