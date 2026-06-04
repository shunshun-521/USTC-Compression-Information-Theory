"""
极化码 BP（置信传播）译码器
基于因子图的消息传递，min-sum 近似，含早停
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _min_sum(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self.info_pos = np.where(self.frozen_bits == 0)[0]
        self._rev = bit_reversal_permutation(N)

    def _cn(self, a, b):
        return _min_sum(a, b, self.alpha)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self._rev]
        n = self.N
        num_iter = self.max_iter
        cn = self._cn

        msg_l = [[None] * (self.n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self.n_stages + 1) for _ in range(num_iter)]
        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        for ind_it in range(num_iter):
            for ind_s in range(self.n_stages):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == self.n_stages - 1:
                    l1_in, l2_in = llr_internal[ind_1], llr_internal[ind_2]
                elif ind_it == 0:
                    l1_in = l2_in = np.zeros(n // 2)
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                r1_out = cn(r1_in, l2_in + r2_in)
                r2_out = cn(r1_in, l1_in) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == self.n_stages - 1:
                    l1_in, l2_in = llr_internal[ind_1], llr_internal[ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                l1_out = cn(l1_in, l2_in + r2_in)
                l2_out = cn(r1_in, l1_in) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            soft = msg_l[ind_it][0]
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[self.info_pos] = np.where(soft[self.info_pos] >= 0, 0, 1).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                return u_hat, ind_it + 1

        return u_hat, num_iter
