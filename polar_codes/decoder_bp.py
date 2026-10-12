"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, g_operation
from encoder import polar_encode
from utils_llr import llr_to_decoder_order


class BPDecoder:
    """BP 译码器（极化因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = int(N)
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_dec = llr_to_decoder_order(llr_ch)
        N, n = self.N, self.n
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_dec
        R[0, :] = 0.0
        R[0, self.frozen] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(i, i + step):
                        La = R[k, j - 1] + L[k + step, j]
                        Lb = L[k, j]
                        L[k, j - 1] = self._f_ms(La, Lb)
                        L[k + step, j - 1] = self._f_ms(R[k, j - 1], L[k, j]) + L[k + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(i, i + step):
                        R[k, j + 1] = self._f_ms(
                            R[k + step, j] + L[k + step, j + 1], R[k, j]
                        )
                        R[k + step, j + 1] = self._f_ms(R[k, j], L[k, j + 1]) + R[k + step, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[0, :] + R[0, :]
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            x_hd = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, x_hd):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
