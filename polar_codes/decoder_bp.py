"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_channel_llrs, _to_frozen_mask
from encoder import polar_encode


_LARGE = 1e6


class BPDecoder:
    """BP 译码器（消息传递索引参考 Sionna PolarBPDecoder）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_mask = _to_frozen_mask(frozen_bits)
        self.frozen_idx = np.where(self.frozen_mask)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum(self, x, y):
        from decoder_sc import f_operation
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_channel_llrs(llr_natural, self.N)
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_l = [[None] * (n + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = _LARGE

        num_iters = num_iter
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(num_iter):
            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
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

                r1_out = self._minsum(r1_in, l2_in + r2_in)
                r2_out = self._minsum(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

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

                l1_out = self._minsum(l1_in, l2_in + r2_in)
                l2_out = self._minsum(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            posterior = msg_l[ind_it][0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        posterior = msg_l[num_iters - 1][0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_mask] = 0
        return u_hat, num_iters


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    bp = BPDecoder(N, frozen_bits)
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(2)
    errors = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_hat, _ = bp.decode(llr)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"BP errors: {errors}/50")
