"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import align_llr_for_decoder
from encoder import polar_encode


def _boxplus_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 1e6

    def _hard_decision_codeword(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = align_llr_for_decoder(llr_ch, self.N)
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max)
        n = self.N
        num_iter = self.max_iter

        msg_l = [[None] * (self.n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self.n_stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        for ind_it in range(num_iter):
            for ind_s in range(self.n_stages):
                ind_range = np.arange(n // 2)
                ind_1 = (ind_range * 2 - np.mod(ind_range, 2 ** ind_s)).astype(int)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2, dtype=np.float64)
                    l2_in = np.zeros(n // 2, dtype=np.float64)
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

                r1_out = _boxplus_min_sum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = (ind_range * 2 - np.mod(ind_range, 2 ** ind_s)).astype(int)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
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

                l1_out = _boxplus_min_sum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            total_llr = msg_l[ind_it][0] + msg_r_in
            u_hat = np.zeros(n, dtype=int)
            for i in range(n):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            if np.array_equal(polar_encode(u_hat), self._hard_decision_codeword(llr_ch)):
                return u_hat, ind_it + 1

        total_llr = msg_l[num_iter - 1][0] + msg_r_in
        u_hat = np.zeros(n, dtype=int)
        for i in range(n):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1
        return u_hat, num_iter
