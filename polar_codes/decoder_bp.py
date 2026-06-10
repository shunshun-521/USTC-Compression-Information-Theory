"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    列 0 为信道端，列 n 为信源端（与 SC 因子图一致）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2**self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[self.frozen_bits, n] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                span = 1 << (s + 1)
                half = span // 2
                for base in range(0, N, span):
                    for k in range(half):
                        i = base + k
                        j = i + half
                        L[i, s + 1] = _f_min_sum(
                            R[i, s] + L[j, s], L[i, s], self.alpha
                        )
                        L[j, s + 1] = _f_min_sum(
                            R[i, s], L[i, s], self.alpha
                        ) + L[j, s]

            for s in range(n - 1, -1, -1):
                span = 1 << (s + 1)
                half = span // 2
                for base in range(0, N, span):
                    for k in range(half):
                        i = base + k
                        j = i + half
                        R[i, s] = _f_min_sum(
                            R[j, s + 1] + L[j, s + 1], R[i, s + 1], self.alpha
                        )
                        R[j, s] = (
                            _f_min_sum(R[i, s + 1], L[i, s + 1], self.alpha)
                            + R[j, s + 1]
                        )

            total = L[:, n] + R[:, n]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[:, n] + R[:, n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
