"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from decoder_sc import f_operation, bit_reversal_indices
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_indices(N)
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = llr_ch[self.br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_ms(
                        R[i, j] + L[i + step, j],
                        L[i, j],
                    )
                    L[i + step, j - 1] = self._f_ms(
                        R[i, j],
                        L[i, j],
                    ) + L[i + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_ms(
                        R[i + step, j] + L[i + step, j],
                        R[i, j - 1],
                    )
                    R[i + step, j] = self._f_ms(
                        R[i, j - 1],
                        L[i, j],
                    ) + R[i + step, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
