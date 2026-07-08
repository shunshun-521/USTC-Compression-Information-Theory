"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图 n+1 列，列 0 为信源端，列 n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        brp = self.brp

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[brp]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    L[i, j - 1] = self._f_ms(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = (
                        self._f_ms(R[i, j], L[i, j]) + L[i + step, j]
                    )

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    R[i, j] = self._f_ms(
                        R[i + step, j] + L[i + step, j], R[i, j - 1]
                    )
                    R[i + step, j] = (
                        self._f_ms(R[i, j - 1], L[i, j]) + R[i + step, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int32)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = np.zeros(N, dtype=np.int32)
            hard_ch[brp] = (llr_ch[brp] < 0).astype(np.int32)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int32)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
