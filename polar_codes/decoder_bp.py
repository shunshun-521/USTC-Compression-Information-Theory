"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import _prepare_channel_llr
from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 1e6

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2**stage)
        ind_2 = ind_1 + 2**stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, num_iters"""
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_channel_llr(llr_ch)
        n = self.n
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iter = 0
        u_hat = np.zeros(self.N, dtype=int)
        msg_l_prev = None

        for it in range(self.max_iter):
            msg_r = [None] * (n + 1)

            for s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0 or msg_l_prev is None:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l1_in = msg_l_prev[ind_1]
                    l2_in = msg_l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = _f_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _f_minsum(r1_in, l1_in, self.alpha) + r2_in
                msg_r[s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            msg_l = [None] * (n + 1)
            for s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_next = msg_l[s + 1]
                    l1_in = l_next[ind_1]
                    l2_in = l_next[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = _f_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _f_minsum(r1_in, l1_in, self.alpha) + l2_in
                msg_l[s] = np.concatenate([l1_out, l2_out])[ind_inv]

            msg_l_prev = msg_l[0]
            num_iter = it + 1
            total = msg_l_prev + msg_r_in
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_orig)):
                break

        total = msg_l_prev + msg_r_in
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iter
