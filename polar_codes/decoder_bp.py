"""
极化码 BP（置信传播）译码器
基于因子图，使用 box-plus 与 min-sum 近似，含早停机制
（消息更新索引参考 Sionna PolarBPDecoder）
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _boxplus(x, y, llr_max=19.3):
    """Check-node update（box-plus）。"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y) + 1e-300)


def _f_minsum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（与 SC 相同的 LLR / 编码比特倒序约定）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_minsum=True):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_minsum = use_minsum
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self._br = bit_reversal_permutation(self.N)

    def _combine(self, a, b):
        if self.use_minsum:
            return _f_minsum(a, b, self.alpha)
        return _boxplus(a, b, self.LLR_MAX)

    def _natural_llr(self, llr_dec):
        inv = np.empty(self.N, dtype=int)
        inv[self._br] = np.arange(self.N)
        return llr_dec[inv]

    def decode(self, llr_ch):
        """返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n_stages

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            # 左 -> 右更新 R
            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
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

                r1_out = self._combine(r1_in, l2_in + r2_in)
                r2_out = self._combine(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            # 右 -> 左更新 L
            for ind_s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n - 1:
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

                l1_out = self._combine(l1_in, l2_in + r2_in)
                l2_out = self._combine(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if msg_l[0][i] >= 0 else 1

            if ind_it % 5 == 4 or ind_it == self.max_iter - 1:
                x_hat = polar_encode(u_hat)
                llr_nat = self._natural_llr(llr_ch)
                hard_ch = (llr_nat < 0).astype(int)
                if np.array_equal(x_hat, hard_ch):
                    break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if msg_l[0][i] >= 0 else 1

        return u_hat, num_iters
