"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 1e6

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def _hard_decision(self, llr_vec):
        u_hat = (llr_vec < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iters = 0
        msg_l_prev = None

        for it in range(self.max_iter):
            num_iters = it + 1
            msg_l = [None] * (self.n + 1)
            msg_r = [None] * (self.n + 1)

            for stage in range(self.n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l_in = msg_l_prev[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _f_min_sum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _f_min_sum(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[stage + 1] = r_out

            for stage in range(self.n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _f_min_sum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _f_min_sum(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[stage] = l_out

            msg_l_prev = msg_l

            u_hat = self._hard_decision(msg_l[0])
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = self._hard_decision(msg_l_prev[0])
        return u_hat, num_iters
