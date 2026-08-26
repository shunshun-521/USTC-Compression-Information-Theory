"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    列 0：信源比特端；列 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits == 1)[0]
        for idx in frozen_idx:
            R[idx, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                span = 1 << (j - 1)
                for block in range(0, N, 2 * span):
                    for i in range(span):
                        idx = block + i
                        s = idx + span
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[s, j], L[idx, j]
                        )
                        L[s, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[s, j]

            for j in range(0, n):
                span = 1 << j
                for block in range(0, N, 2 * span):
                    for i in range(span):
                        idx = block + i
                        s = idx + span
                        R[idx, j + 1] = self._f_ms(
                            R[s, j] + L[s, j + 1], R[idx, j]
                        )
                        R[s, j + 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + R[s, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
