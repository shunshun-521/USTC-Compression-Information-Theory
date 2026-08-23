"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0=信源端，列 n=信道端）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br_inv = np.argsort(bit_reversal_permutation(N))

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br_inv]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr
        R[0, ~self.frozen_bits] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << (stage + 1)
                half = 1 << stage
                for i in range(0, N, step):
                    for j in range(half):
                        top = i + j
                        bot = top + half
                        L[stage, top] = self._f(
                            R[stage + 1, top] + L[stage + 1, bot],
                            L[stage + 1, top],
                        )
                        L[stage, bot] = self._f(
                            R[stage + 1, top],
                            L[stage + 1, top],
                        ) + L[stage + 1, bot]

            for stage in range(n):
                step = 1 << (stage + 1)
                half = 1 << stage
                for i in range(0, N, step):
                    for j in range(half):
                        top = i + j
                        bot = top + half
                        R[stage + 1, top] = self._f(
                            R[stage, bot] + L[stage + 1, bot],
                            R[stage, top],
                        )
                        R[stage + 1, bot] = self._f(
                            R[stage, top],
                            L[stage + 1, top],
                        ) + R[stage, bot]

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

            num_iters = it
            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

        return u_hat, num_iters
