"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（Sionna 因子图索引，与 polar_encode 配套）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6
        self._precompute_indices()

    def _precompute_indices(self):
        self.stage_indices = []
        for ind_s in range(self.n):
            ind_range = np.arange(self.N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self.stage_indices.append((ind_1, ind_2, ind_inv))

    def _boxplus(self, a, b):
        return self.alpha * f_operation(a, b)

    @staticmethod
    def _encode_nobr(u):
        """无比特倒序编码，用于 BP 早停校验。"""
        u = np.asarray(u, dtype=int).copy()
        N = len(u)
        block = N
        while block > 1:
            half = block // 2
            for start in range(0, N, block):
                for i in range(start, start + half):
                    u[i] ^= u[i + half]
            block = half
        return u

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        n = self.n
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.LARGE

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self.stage_indices[ind_s]

                if ind_s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2, dtype=np.float64)
                    l2_in = np.zeros(N // 2, dtype=np.float64)
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

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self.stage_indices[ind_s]

                if ind_s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
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
                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            total = msg_l[0]
            u_hat[:] = 0
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = msg_l[0]
        u_hat[:] = 0
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
