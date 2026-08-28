"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        self.info_idx = np.where(~self.frozen_bits.astype(bool))[0]
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j - 1]
                    Ris = R[i + s, j - 1]
                    L[i, j - 1] = _f_min_sum(Ri + Lis, Li, alpha)
                    L[i + s, j - 1] = _f_min_sum(Ri, Li, alpha) + Lis

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    Li = L[i, j + 1]
                    Lis = L[i + s, j + 1]
                    R[i, j + 1] = _f_min_sum(Ris + Lis, Ri, alpha)
                    R[i + s, j + 1] = _f_min_sum(Ri, Li, alpha) + Ris

            num_iters = it
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
