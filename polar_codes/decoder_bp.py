"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L0, R0):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L0[i] + R0[i]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_bits == 1] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        ip = i + step
                        L[stage, i] = self._f(
                            R[stage, i] + L[stage + 1, ip], L[stage + 1, i]
                        )
                        L[stage, ip] = self._f(R[stage, i], L[stage + 1, i]) + L[
                            stage + 1, ip
                        ]

            for stage in range(0, n):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        ip = i + step
                        R[stage + 1, i] = self._f(
                            R[stage, ip] + L[stage + 1, ip], R[stage, i]
                        )
                        R[stage + 1, ip] = self._f(R[stage, i], L[stage + 1, i]) + R[
                            stage, ip
                        ]

            u_hat = self._hard_decision(L[0, :], R[0, :])
            if self._early_stop(u_hat, llr_ch):
                break

        return self._hard_decision(L[0, :], R[0, :]), num_iters
