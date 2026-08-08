"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器（参考 Sionna PolarBPDecoder 结构，min-sum 近似）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._llr_max = 19.3

    def _min_sum(self, x, y):
        x = np.clip(x, -self._llr_max, self._llr_max)
        y = np.clip(y, -self._llr_max, self._llr_max)
        return self.alpha * f_operation(x, y)

    def _stage_indices(self, ind_s):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_nat[br]
        num_iter = self.max_iter

        msg_l = [[None] * (self.n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (self.n_stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        for ind_it in range(num_iter):
            # 左到右更新 R 消息
            for ind_s in range(self.n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
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

                r1_out = self._min_sum(r1_in, l2_in + r2_in)
                r2_out = self._min_sum(r1_in, l1_in) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            # 右到左更新 L 消息
            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

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

                l1_out = self._min_sum(l1_in, l2_in + r2_in)
                l2_out = self._min_sum(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            # 早停检查（每 5 次迭代检查一次以降低开销）
            if (ind_it + 1) % 5 == 0 or ind_it == self.max_iter - 1:
                total = msg_l[ind_it][0]
                u_hat = np.zeros(self.N, dtype=np.int8)
                u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(np.int8)
                x_hat = polar_encode(u_hat)
                hard_ch = (llr_nat < 0).astype(np.int8)
                if np.array_equal(x_hat, hard_ch):
                    num_iter = ind_it + 1
                    break

        total = msg_l[num_iter - 1][0]
        u_hat = np.zeros(self.N, dtype=np.int8)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(np.int8)
        return u_hat, num_iter
