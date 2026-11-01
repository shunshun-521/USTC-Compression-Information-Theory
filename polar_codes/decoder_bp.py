"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = frozen_bits.copy()
        else:
            self.frozen = frozen_bits.astype(int) != 0
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        br = bit_reversal_permutation(N)
        llr = llr_ch[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_ms(
                        R[i, j] + L[i + s, j + 1], L[i, j + 1]
                    )
                    L[i + s, j - 1] = self._f_ms(R[i, j], L[i, j + 1]) + L[i + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_ms(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
