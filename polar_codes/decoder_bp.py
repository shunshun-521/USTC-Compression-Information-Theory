"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

    def _cn_update(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        if self.alpha >= 1.0:
            return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数。"""
        n = self.n
        N = self.N
        llr_ch = -np.asarray(llr_ch, dtype=np.float64)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_indices] = self.llr_max

        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]
        num_iters = self.max_iter
        u_hat_full = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
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

                r1_out = self._cn_update(r1_in, l2_in + r2_in)
                r2_out = self._cn_update(r1_in, l1_in) + r2_in
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n - 1:
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

                l1_out = self._cn_update(l1_in, l2_in + r2_in)
                l2_out = self._cn_update(r1_in, l1_in) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            for i in self.info_indices:
                total = msg_l[ind_it][0][i] + msg_r_in[i]
                u_hat_full[i] = 0 if total > 0 else 1

            x_hat = polar_encode(u_hat_full)
            hard_bits = (llr_ch > 0).astype(int)
            if np.array_equal(x_hat, hard_bits):
                num_iters = ind_it + 1
                break

        for i in self.info_indices:
            total = msg_l[num_iters - 1][0][i] + msg_r_in[i]
            u_hat_full[i] = 0 if total > 0 else 1

        return u_hat_full, num_iters
