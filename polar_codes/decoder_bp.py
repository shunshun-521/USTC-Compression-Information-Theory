"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation_min_sum

LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.brp = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        return self.alpha * f_operation_min_sum(x, y)

    def _hard_bits(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.brp]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + s, j],
                            L[idx, j],
                        )
                        L[idx + s, j - 1] = (
                            self._f_ms(R[idx, j], L[idx, j])
                            + L[idx + s, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                        )
                        R[idx + s, j + 1] = (
                            self._f_ms(R[idx, j], L[idx, j + 1])
                            + R[idx + s, j]
                        )

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
