"""
极化码 BP（置信传播）译码器
基于 Sionna/Arikan 因子图结构，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


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
        self.llr_max = 19.3

    def _minsum_f(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return self.alpha * f_operation(x, y)

    def _stage_indices(self, ind_s):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
        ind_2 = ind_1 + 2**ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

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

                r1_out = self._minsum_f(r1_in, l2_in + r2_in)
                r2_out = self._minsum_f(r1_in, l1_in) + r2_in
                msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

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

                l1_out = self._minsum_f(l1_in, l2_in + r2_in)
                l2_out = self._minsum_f(r1_in, l1_in) + l2_in
                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_soft = msg_l[0]
            u_hat[:] = 0
            u_hat[self.info_pos] = (u_soft[self.info_pos] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        u_soft = msg_l[0]
        u_hat[:] = 0
        u_hat[self.info_pos] = (u_soft[self.info_pos] < 0).astype(int)
        return u_hat, num_iters
