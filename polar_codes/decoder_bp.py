"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _permute_channel_llr

LLR_MAX = 19.3


def _minsum_f(a, b, alpha):
    """min-sum 近似 f 运算，带修正因子 alpha"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _stage_indices(N, stage):
    """计算第 stage 层蝶形单元的上下分支索引"""
    half = N // 2
    ind_range = np.arange(half)
    ind_1 = ind_range * 2 - np.mod(ind_range, 1 << stage)
    ind_2 = ind_1 + (1 << stage)
    ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
    return ind_1, ind_2, ind_inv


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = _permute_channel_llr(llr_ch, self.N)
        llr = np.clip(llr, -LLR_MAX, LLR_MAX)

        N = self.N
        n = self.n
        f = lambda a, b: _minsum_f(a, b, self.alpha)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = LLR_MAX

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            for stage in range(n):
                ind_1, ind_2, ind_inv = _stage_indices(N, stage)

                if stage == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l_prev = msg_l[stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = f(r1_in, l2_in + r2_in)
                r2_out = f(r1_in, l1_in) + r2_in
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = _stage_indices(N, stage)

                if stage == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l_prev = msg_l[stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = f(l1_in, l2_in + r2_in)
                l2_out = f(r1_in, l1_in) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            soft = msg_l[0]
            u_hat = (soft < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        soft = msg_l[0]
        u_hat = (soft < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
