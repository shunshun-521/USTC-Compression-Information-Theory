"""
极化码 BP（置信传播）译码器
基于因子图，使用 box-plus 运算，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（参考 Arikan BP / Sionna 实现）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.llr_max = 19.3
        self.alpha = alpha  # 保留参数，box-plus 中未使用

    def _boxplus(self, x, y):
        """Check-node 更新（box-plus，数值稳定）"""
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y) + 1e-300)

    def _f_min_sum(self, x, y):
        """min-sum 近似（可选）"""
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        num_iter = self.max_iter
        msg_l = [[None] * (self.n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self.n_stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iters_done = num_iter
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(num_iter):
            # 左到右更新 R
            for ind_s in range(self.n_stages):
                ind_range = np.arange(self.N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
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

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            # 右到左更新 L
            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_range = np.arange(self.N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s
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

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            # 硬判决与早停
            soft_u = msg_l[ind_it][0]
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[self.info_pos] = np.where(soft_u[self.info_pos] > 0, 0, 1)
            u_hat[self.frozen_pos] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters_done = ind_it + 1
                break

        soft_u = msg_l[num_iters_done - 1][0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_pos] = np.where(soft_u[self.info_pos] > 0, 0, 1)
        u_hat[self.frozen_pos] = 0
        return u_hat, num_iters_done
