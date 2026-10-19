"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_bits == 1] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for lvl in reversed(range(n)):
                bs = 1 << lvl
                for j in range(0, N, 2 * bs):
                    for i in range(bs):
                        a, b = j + i, j + i + bs
                        L[lvl, a] = self._f_ms(R[lvl, a] + L[lvl + 1, b], L[lvl + 1, a])
                        L[lvl, b] = self._f_ms(R[lvl, a], L[lvl + 1, a]) + L[lvl + 1, b]

            for lvl in range(n):
                bs = 1 << lvl
                for j in range(0, N, 2 * bs):
                    for i in range(bs):
                        a, b = j + i, j + i + bs
                        R[lvl + 1, a] = self._f_ms(
                            R[lvl + 1, b] + L[lvl + 1, b], R[lvl, a]
                        )
                        R[lvl + 1, b] = self._f_ms(R[lvl, a], L[lvl + 1, a]) + R[lvl + 1, b]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            x_hd = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hd):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
