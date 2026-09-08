"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _prepare_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = 1 if a > 0 else (-1 if a < 0 else 1)
    sb = 1 if b > 0 else (-1 if b < 0 else 1)
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 2 ** layer
                for i in range(0, N, 2 * step):
                    L[i, layer] = _f_min_sum(
                        R[i, layer + 1] + L[i + step, layer + 1],
                        L[i, layer + 1],
                        self.alpha,
                    )
                    L[i + step, layer] = _f_min_sum(
                        R[i, layer + 1], L[i, layer + 1], self.alpha
                    ) + L[i + step, layer + 1]

            for layer in range(0, n):
                step = 2 ** layer
                for i in range(0, N, 2 * step):
                    R[i, layer + 1] = _f_min_sum(
                        R[i + step, layer + 1] + L[i + step, layer + 1],
                        R[i, layer],
                        self.alpha,
                    )
                    R[i + step, layer + 1] = _f_min_sum(
                        R[i, layer], L[i, layer + 1], self.alpha
                    ) + R[i + step, layer + 1]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
