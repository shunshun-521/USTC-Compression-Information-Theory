"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha

        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype != bool:
            self.frozen_mask = frozen_bits.astype(bool)
        else:
            self.frozen_mask = frozen_bits.copy()

        self.large = 1e6

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        from decoder_sc import bit_reversal_permutation

        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    li = slice(i, i + s)
                    ri = slice(i + s, i + 2 * s)
                    L[li, j] = self._f(
                        R[li, j + 1] + L[ri, j + 1],
                        L[li, j + 1],
                    )
                    L[ri, j] = self._f(R[li, j + 1], L[li, j + 1]) + L[ri, j + 1]

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    li = slice(i, i + s)
                    ri = slice(i + s, i + 2 * s)
                    R[li, j + 1] = self._f(
                        R[ri, j + 1] + L[ri, j + 1],
                        R[li, j],
                    )
                    R[ri, j + 1] = self._f(R[li, j], L[li, j + 1]) + R[ri, j + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
