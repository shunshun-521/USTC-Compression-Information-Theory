"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
（参考 Sionna / Arikan BP 因子图索引）
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e9

    def _boxplus(self, x, y):
        sign = np.sign(x) * np.sign(y)
        mag = np.minimum(np.abs(x), np.abs(y))
        return self.alpha * sign * mag

    def _stage_indices(self, ind_s):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        n_stages = self.n
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        for idx in self.frozen_set:
            msg_r_in[idx] = self.LARGE

        msg_l_hist = []
        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l = [None] * (n_stages + 1)
            msg_r = [None] * (n_stages + 1)

            # 从左到右更新 R 消息
            for ind_s in range(n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1), dtype=np.float64)
                    l2_in = np.zeros(len(ind_2), dtype=np.float64)
                else:
                    l_in = msg_l_hist[ind_it - 1][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                r_out = np.empty(N, dtype=np.float64)
                r_concat = np.concatenate([r1_out, r2_out])
                r_out[ind_inv] = r_concat
                msg_r[ind_s + 1] = r_out

            # 从右到左更新 L 消息
            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n_stages - 1:
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

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                l_out = np.empty(N, dtype=np.float64)
                l_concat = np.concatenate([l1_out, l2_out])
                l_out[ind_inv] = l_concat
                msg_l[ind_s] = l_out

            msg_l_hist.append(msg_l)

            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if msg_l[0][i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break  # 早停

        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if msg_l_hist[-1][0][i] >= 0 else 1

        return u_hat, num_iters
