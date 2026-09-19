"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import _bitrev_indices, polar_encode


def _min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


def _precompute_stage_indices(n):
    """预计算各 stage 的蝶形索引（参考 Sionna PolarBPDecoder）。"""
    n_stages = int(np.log2(n))
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
    """BP 译码器（因子图 min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._llr_max = 1e6
        self._stage_ind_1, self._stage_ind_2, self._stage_ind_inv = _precompute_stage_indices(N)

    def _bp_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        n_stages = self.n
        msg_l_iter = [None] * (n_stages + 1)
        msg_r_iter = [None] * (n_stages + 1)
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        for ind_s in range(n_stages):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == n_stages - 1:
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

            r1_out = _min_sum(r1_in, l2_in + r2_in, self.alpha)
            r2_out = _min_sum(r1_in, l1_in, self.alpha) + r2_in

            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(n_stages - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == n_stages - 1:
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

            l1_out = _min_sum(l1_in, l2_in + r2_in, self.alpha)
            l2_out = _min_sum(r1_in, l1_in, self.alpha) + l2_in

            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_nat[_bitrev_indices(self.N)]
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        msg_l_prev = None
        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)
        hard_ch = (llr_nat < 0).astype(int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l_iter, _ = self._bp_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter

            soft = msg_l_iter[0]
            u_hat = np.where(soft >= 0, 0, 1)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                break

        soft = msg_l_prev[0]
        u_hat = np.where(soft >= 0, 0, 1)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
