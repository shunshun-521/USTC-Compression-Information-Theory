"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus(x, y):
    """精确 box-plus（sum-product 校验节点）"""
    x = np.clip(x, -19.3, 19.3)
    y = np.clip(y, -19.3, 19.3)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _sign_llr(x):
    return np.where(x >= 0, 1.0, -1.0)


def _f_min_sum(La, Lb, alpha):
    return alpha * _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


class BPDecoder:
    """BP 译码器（参考 Arikan/Sionna 因子图消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._llr_max = 19.3

    def _f(self, x, y):
        return _f_min_sum(x, y, self.alpha)

    def _stage_indices(self, stage):
        """Sionna 风格的阶段节点索引"""
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n_stages
        N = self.N

        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]

        msg_r_in = np.zeros(N)
        msg_r_in[self.frozen_pos] = self._llr_max

        num_iters = self.max_iter
        for ind_it in range(self.max_iter):
            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
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

                r1_out = self._f(r1_in, l2_in + r2_in)
                r2_out = self._f(r1_in, l1_in) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
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

                l1_out = self._f(l1_in, l2_in + r2_in)
                l2_out = self._f(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            u_hat = np.zeros(N, dtype=int)
            llr_left = msg_l[ind_it][0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if llr_left[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break
        else:
            llr_left = msg_l[self.max_iter - 1][0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or llr_left[i] >= 0 else 1

        return u_hat, num_iters
