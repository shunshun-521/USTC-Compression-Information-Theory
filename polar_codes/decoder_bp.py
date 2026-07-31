"""
极化码 BP（置信传播）译码器
基于因子图 min-sum 近似（参考 Sionna/Arikan BP），含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import channel_llr_to_decoder, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 19.3

    def _boxplus_minsum(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return self.alpha * f_operation(x, y)

    def _decode_bp(self, llr_ch, num_iter):
        n = self.N
        stages = self.n_stages

        msg_l = [[None] * (stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        for ind_it in range(num_iter):
            for ind_s in range(stages):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2)
                    l2_in = np.zeros(n // 2)
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

                r1_out = self._boxplus_minsum(r1_in, l2_in + r2_in)
                r2_out = self._boxplus_minsum(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == stages - 1:
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

                l1_out = self._boxplus_minsum(l1_in, l2_in + r2_in)
                l2_out = self._boxplus_minsum(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

        u_llr = msg_l[num_iter - 1][0][self.info_idx]
        u_hat = np.zeros(n, dtype=int)
        u_hat[self.info_idx] = (u_llr < 0).astype(int)
        return u_hat

    def decode(self, llr_ch):
        llr_ch = channel_llr_to_decoder(llr_ch)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            u_hat = self._decode_bp(llr_ch, it)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break
        else:
            u_hat = self._decode_bp(llr_ch, self.max_iter)

        return u_hat, num_iters
