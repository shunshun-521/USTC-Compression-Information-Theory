"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（min-sum + 归一化因子 alpha）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self._br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        llr_ch: 蝶形顺序 LLR（channel_llr_for_decoder 输出）。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        Ld = np.zeros((n + 1, N), dtype=np.float64)
        Rd = np.zeros((n + 1, N), dtype=np.float64)
        Ld[n] = llr_ch
        Rd[0] = 0.0
        Rd[0, self.frozen_bits] = self._large

        num_iters = 0
        hard_air = (llr_ch[self._br] < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n - 1, -1, -1):
                bs = 1 << s
                for i in range(0, N, bs * 2):
                    for j in range(bs):
                        u = i + j
                        Ld[s, u] = self._f_ms(
                            Rd[s, u] + Ld[s + 1, u + bs], Ld[s + 1, u]
                        )
                        Ld[s, u + bs] = (
                            self._f_ms(Rd[s, u], Ld[s + 1, u]) + Ld[s + 1, u + bs]
                        )

            for s in range(1, n + 1):
                bs = 1 << (s - 1)
                for i in range(0, N, bs * 2):
                    for j in range(bs):
                        u = i + j
                        Rd[s, u] = self._f_ms(
                            Rd[s - 1, u + bs] + Ld[s, u + bs], Rd[s - 1, u]
                        )
                        Rd[s, u + bs] = (
                            self._f_ms(Rd[s - 1, u], Ld[s, u]) + Rd[s - 1, u + bs]
                        )

            total = Ld[0] + Rd[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            if np.array_equal(polar_encode(u_hat), hard_air):
                break

        total = Ld[0] + Rd[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
