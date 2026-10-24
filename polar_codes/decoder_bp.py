"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（因子图 n+1 列，列 0 为信源端，列 n 为信道端）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        llr_int = llr_ch[bit_reversal_permutation(N)]
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_int
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    Li = i
                    Ri = i + step
                    L[Li, j - 1] = self._f_ms(
                        R[Li, j] + L[Ri, j + 1], L[Li, j + 1]
                    )
                    L[Ri, j - 1] = self._f_ms(R[Li, j], L[Li, j + 1]) + L[Ri, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    Li = i
                    Ri = i + step
                    R[Li, j + 1] = self._f_ms(
                        R[Ri, j] + L[Ri, j + 1], R[Li, j]
                    )
                    R[Ri, j + 1] = self._f_ms(R[Li, j], L[Li, j + 1]) + R[Ri, j]

            u_hat = np.zeros(N, dtype=np.int8)
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_int < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        u_hat = np.where(L[:, 0] + R[:, 0] >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen] = 0
        return u_hat, num_iters
