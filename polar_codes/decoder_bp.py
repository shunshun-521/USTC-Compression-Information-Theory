"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode

LARGE = 1e7


class BPDecoder:
    """
    BP 译码器（分层因子图上的 min-sum 消息传递）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_idx:
            R[idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for phi in range(0, N, 2 * step):
                    for omega in range(step):
                        a = phi + omega
                        b = a + step
                        L[a, lam] = self._f_ms(
                            R[a, lam + 1] + L[b, lam + 1], L[a, lam + 1]
                        )
                        L[b, lam] = self._f_ms(R[a, lam + 1], L[a, lam + 1]) + L[b, lam + 1]

            for lam in range(0, n):
                step = 1 << lam
                for phi in range(0, N, 2 * step):
                    for omega in range(step):
                        a = phi + omega
                        b = a + step
                        R[a, lam + 1] = self._f_ms(
                            R[b, lam] + L[b, lam + 1], R[a, lam]
                        )
                        R[b, lam + 1] = self._f_ms(R[a, lam], L[b, lam + 1]) + R[b, lam]

            for idx in self.frozen_idx:
                R[idx, 0] = LARGE
                L[idx, 0] = LARGE

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
