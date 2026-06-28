"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（Permuted factor graph 上的 flooding BP）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        a = np.clip(a, -self.LLR_MAX, self.LLR_MAX)
        b = np.clip(b, -self.LLR_MAX, self.LLR_MAX)
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2**stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.rev]

        n = self.n
        N = self.N
        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            for stage in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l_in = msg_l[ind_it - 1][stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._f_ms(r1_in, l2_in + r2_in)
                r2_out = self._f_ms(r1_in, l1_in) + r2_in
                msg_r[ind_it][stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[ind_it][stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = self._f_ms(l1_in, l2_in + r2_in)
                l2_out = self._f_ms(r1_in, l1_in) + l2_in
                msg_l[ind_it][stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            llr_total = msg_l[ind_it][0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if llr_total[i] >= 0 else 1

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = ind_it + 1
                break

        llr_total = msg_l[num_iters - 1][0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_total[i] >= 0 else 1

        return u_hat, num_iters
