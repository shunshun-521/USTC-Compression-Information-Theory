"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def ms_boxplus(x, y, alpha):
    """min-sum 近似的 boxplus（check-node）运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._br = bit_reversal_permutation(N)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_natural = llr_ch.copy()
        # 与含比特倒序的编码器配套：信道 LLR 需做倒序置换
        llr_ch = llr_ch[self._br]
        n = self.n
        N = self.N
        alpha = self.alpha

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX
        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            msg_l = [None] * (n + 1)
            msg_r = [None] * (n + 1)

            # 左到右更新 R 消息
            for s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif msg_l[s + 1] is None:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l_in = msg_l[s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = ms_boxplus(r1_in, l2_in + r2_in, alpha)
                r2_out = ms_boxplus(r1_in, l1_in, alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[s + 1] = r_out

            # 右到左更新 L 消息
            for s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = ms_boxplus(l1_in, l2_in + r2_in, alpha)
                l2_out = ms_boxplus(r1_in, l1_in, alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[s] = l_out

            llr_total = msg_l[0]
            for i in range(N):
                u_hat[i] = 0 if llr_total[i] > 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_natural < 0).astype(int)
            num_iters = it + 1
            if np.array_equal(x_hat, hard):
                break

        return u_hat, num_iters
