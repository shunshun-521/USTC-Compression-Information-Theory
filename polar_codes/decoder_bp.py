"""
极化码 BP（置信传播）译码器
基于因子图（Sionna/Arikan 调度），支持 min-sum 与精确 box-plus，含早停
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器：因子图 n+1 列，每列 N 个节点。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_min_sum = use_min_sum
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 19.3

    def _boxplus(self, x, y):
        x = np.clip(np.asarray(x, dtype=np.float32), -self.llr_max, self.llr_max)
        y = np.clip(np.asarray(y, dtype=np.float32), -self.llr_max, self.llr_max)
        if self.use_min_sum:
            return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))
        return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def _hard_decide(self, soft):
        u_hat = np.zeros(self.N, dtype=int)
        for i in self.info_idx:
            u_hat[i] = 0 if soft[i] > 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。

        参数 llr_ch：信道 LLR，LLR>0 倾向 bit 0。
        返回 (u_hat, num_iters)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        n = self.n
        N = self.N
        num_iters = 0
        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r_in = np.zeros(N, dtype=np.float32)
        msg_r_in[self.frozen_idx] = self.llr_max

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            for stage in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)
                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1), dtype=np.float32)
                    l2_in = np.zeros(len(ind_2), dtype=np.float32)
                else:
                    l_prev = msg_l[ind_it - 1][stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[ind_it][stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                msg_r[ind_it][stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)
                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_prev = msg_l[ind_it][stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[ind_it][stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                msg_l[ind_it][stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            soft = msg_l[ind_it][0]
            u_hat = self._hard_decide(soft)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return self._hard_decide(msg_l[num_iters - 1][0]), num_iters
