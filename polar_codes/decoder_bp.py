"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（Sionna/Arikan 因子图消息传递，min-sum 近似）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _boxplus(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n_stages = self.n_stages
        num_iter = self.max_iter
        llr_batch = llr_ch.reshape(1, N)

        msg_r_in = np.zeros((1, N), dtype=np.float64)
        msg_r_in[0, self.frozen] = self._large

        msg_l = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(num_iter)]

        num_iters = num_iter
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(num_iter):
            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in, l2_in = llr_batch[:, ind_1], llr_batch[:, ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros((1, N // 2))
                    l2_in = np.zeros((1, N // 2))
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in, l2_in = l_in[:, ind_1], l_in[:, ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[:, ind_1], msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in, r2_in = r_in[:, ind_1], r_in[:, ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate(
                    [r1_out, r2_out], axis=1
                )[:, ind_inv]

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n_stages - 1:
                    l1_in, l2_in = llr_batch[:, ind_1], llr_batch[:, ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in, l2_in = l_in[:, ind_1], l_in[:, ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[:, ind_1], msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in, r2_in = r_in[:, ind_1], r_in[:, ind_2]

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate(
                    [l1_out, l2_out], axis=1
                )[:, ind_inv]

            u_soft = msg_l[ind_it][0][0]
            info_idx = np.where(~self.frozen)[0]
            u_hat[info_idx] = (u_soft[info_idx] <= 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = ind_it + 1
                break

        return u_hat, num_iters


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.default_rng(0).integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.001)
    bp = BPDecoder(N, frozen)
    u_hat, iters = bp.decode(llr)
    print("BP decode ok", np.array_equal(u_hat, u), "iters", iters)
