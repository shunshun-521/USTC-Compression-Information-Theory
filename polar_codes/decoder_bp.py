"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_min_sum(x, y, alpha):
    x = np.clip(x, -50.0, 50.0)
    y = np.clip(y, -50.0, 50.0)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（参考 Sionna PolarBPDecoder 结构）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.llr_max = 19.3

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l = [None] * (self.n_stages + 1)
        msg_r = [None] * (self.n_stages + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            for ind_s in range(self.n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
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

                r1_out = _boxplus_min_sum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + r2_in

                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == self.n_stages - 1:
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

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            info_pos = np.where(~self.frozen_bits)[0]
            soft = msg_l[0][info_pos]
            u_hat[:] = 0
            u_hat[info_pos] = (soft < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat[self.br], hard_ch):
                num_iters = ind_it + 1
                break

        info_pos = np.where(~self.frozen_bits)[0]
        soft = msg_l[0][info_pos]
        u_hat[:] = 0
        u_hat[info_pos] = (soft < 0).astype(int)
        return u_hat, num_iters
