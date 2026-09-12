"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum / box-plus 近似，含早停机制
（算法参考 Sionna PolarBPDecoder，适配本项目的 LLR 约定）
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    """Check-node 更新（log-domain box-plus）。"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _precompute_stage_indices(n):
    """预计算各 stage 的蝶形索引。"""
    n_stages = int(math.log2(n))
    ind_range = np.arange(n // 2)
    stage_ind_1 = []
    stage_ind_2 = []
    stage_ind_inv = []
    for ind_s in range(n_stages):
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        stage_ind_1.append(ind_1)
        stage_ind_2.append(ind_2)
        stage_ind_inv.append(ind_inv)
    return stage_ind_1, stage_ind_2, stage_ind_inv


class BPDecoder:
    """BP 译码器（分层因子图，box-plus 近似）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3
        self._stage_ind_1, self._stage_ind_2, self._stage_ind_inv = (
            _precompute_stage_indices(N)
        )

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        """单次 BP 迭代（左→右 + 右→左）。"""
        msg_l_iter = [None] * (self.n_stages + 1)
        msg_r_iter = [None] * (self.n_stages + 1)
        zeros_half = np.zeros(self.N // 2)

        for ind_s in range(self.n_stages):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self.n_stages - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif ind_it == 0:
                l1_in = zeros_half
                l2_in = zeros_half
            else:
                l_in = msg_l_prev[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = _boxplus(r1_in, l2_in + r2_in, self.llr_max)
            r2_out = _boxplus(r1_in, l1_in, self.llr_max) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(self.n_stages - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self.n_stages - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l_iter[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = _boxplus(l1_in, l2_in + r2_in, self.llr_max)
            l2_out = _boxplus(r1_in, l1_in, self.llr_max) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        """
        主译码函数，返回 (u_hat, num_iters)。
        输入 LLR 约定：正值倾向比特 0（与 channel.py 一致）。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        llr_internal = llr_ch

        msg_r_in = np.zeros(self.N)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l_prev = None
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(
                ind_it, llr_internal, msg_l_prev, msg_r_in
            )
            msg_l_prev = msg_l_iter

            soft = msg_l_iter[0]
            for i in self.info_pos:
                u_hat[i] = 0 if soft[i] > 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = ind_it + 1
                break

        soft = msg_l_prev[0]
        for i in self.info_pos:
            u_hat[i] = 0 if soft[i] > 0 else 1
        for i in self.frozen_pos:
            u_hat[i] = 0

        return u_hat, num_iters
