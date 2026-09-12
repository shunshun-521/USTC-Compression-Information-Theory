"""
极化码 BP（置信传播）译码器
基于因子图迭代软信息传播（SCAN 型调度 + min-sum），含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(self.N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_llrs_scan(self, l, L, beta_prev, n):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = self._f_ms(L[j, s], L[j + branch_size, s])
                else:
                    b_val = beta_prev[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], b_val
                    )

    def _update_bits(self, l, beta, n):
        if l < self.N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    beta[j - branch_size, s - 1] = (
                        beta[j, s] ^ beta[j - branch_size, s]
                    )
                    beta[j, s - 1] = beta[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        n = self.n
        L = np.zeros((self.N, n + 1), dtype=np.float64)
        beta = np.zeros((self.N, n + 1), dtype=np.int32)
        u_hat = np.zeros(self.N, dtype=int)
        num_iters = self.max_iter

        for iteration in range(1, self.max_iter + 1):
            beta_prev = beta.copy()
            L[:, 0] = llr_ch.copy()

            for phi in range(self.N):
                l = bit_reversed(phi, n)
                self._update_llrs_scan(l, L, beta_prev, n)

                if l in self.frozen_set:
                    u_hat[l] = 0
                    beta[l, n] = 0
                else:
                    u_hat[l] = 0 if L[l, n] >= 0 else 1
                    beta[l, n] = u_hat[l]

                self._update_bits(l, beta, n)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat.copy(), iteration

        return u_hat.copy(), num_iters
