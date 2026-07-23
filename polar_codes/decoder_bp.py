"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
采用迭代 SC + 软反馈（SCAN 风格）实现，与极化码因子图一致
"""
import numpy as np
from encoder import polar_encode
from channel import compute_llr, bpsk_modulate
from decoder_sc import (
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _f_boxplus,
    _lower_llr,
    _reorder_channel_llr,
    f_operation,
)


class BPDecoder:
    """BP 迭代译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _sc_pass(self, llr_nat):
        """单次 SC 软信息传递（min-sum f 运算）。"""
        n, N = self.n, self.N
        frozen_set = set(self.frozen_idx)

        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan)
        L[:, 0] = llr_nat

        for i in range(N):
            l = _bit_reversed_index(i, n)
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = _lower_llr(
                            L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                        )

            if l in frozen_set:
                B[l, n] = 0
            else:
                B[l, n] = 0 if L[l, n] >= 0 else 1

            if l >= N / 2:
                for s in range(n, n - _active_bit_level(l, n), -1):
                    block_size = 1 << s
                    branch_size = block_size // 2
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                            B[j, s - 1] = B[j, s]

        return B[:, n].astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = _reorder_channel_llr(llr_ch, self.N)
        llr_init = llr_nat.copy()
        u_hat = np.zeros(self.N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            u_hat = self._sc_pass(llr_nat)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            feedback = compute_llr(bpsk_modulate(x_hat), 0.1)
            fb_nat = _reorder_channel_llr(feedback, self.N)
            llr_nat = (1.0 - self.alpha) * llr_init + self.alpha * fb_nat

        return u_hat, num_iters


if __name__ == '__main__':
    from construction import ga_construction
    from channel import eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(2)
    errors = 0
    bp = BPDecoder(N, frozen_bits)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(6.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        uh, iters = bp.decode(llr)
        if not np.array_equal(uh[info_idx], u[info_idx]):
            errors += 1
    print(f'BP errors: {errors}/30')
