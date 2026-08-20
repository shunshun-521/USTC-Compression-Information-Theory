"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（极化码因子图，stage-combined min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype != bool:
            frozen_bits = frozen_bits.astype(bool)
        self.frozen_bits = frozen_bits
        self.frozen_idx = np.where(frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e8

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        llr = llr_ch[self.rev].copy()

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                Ni = 2**stage
                for i in range(0, N, 2 * Ni):
                    for j in range(Ni):
                        L[stage, i + j] = self._f_ms(
                            L[stage + 1, i + j],
                            L[stage + 1, i + j + Ni] + R[stage + 1, i + j + Ni],
                        )
                        L[stage, i + j + Ni] = L[stage + 1, i + j + Ni] + self._f_ms(
                            L[stage + 1, i + j], R[stage + 1, i + j]
                        )

            for stage in range(n):
                Ni = 2**stage
                for i in range(0, N, 2 * Ni):
                    for j in range(Ni):
                        R[stage + 1, i + j] = self._f_ms(
                            R[stage, i + j],
                            L[stage + 1, i + j + Ni] + R[stage + 1, i + j + Ni],
                        )
                        R[stage + 1, i + j + Ni] = R[
                            stage + 1, i + j + Ni
                        ] + self._f_ms(R[stage, i + j], L[stage + 1, i + j])

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters
