"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_minsum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
  因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        inv_br = np.argsort(bit_reversal_permutation(self.N))
        llr_aligned = llr_ch[inv_br]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_aligned
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    Li = i
                    Li_s = i + step
                    Rij = R[Li, j - 1] if j > 0 else 0.0
                    L[i, j - 1] = _boxplus_minsum(
                        Rij + L[Li_s, j], L[Li, j], self.alpha
                    )
                    L[Li_s, j - 1] = _boxplus_minsum(R[Li, j - 1], L[Li, j], self.alpha) + L[
                        Li_s, j
                    ]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    Li = i
                    Li_s = i + step
                    R[Li, j + 1] = _boxplus_minsum(
                        R[Li_s, j] + L[Li_s, j + 1], R[Li, j], self.alpha
                    )
                    R[Li_s, j + 1] = (
                        _boxplus_minsum(R[Li, j], L[Li, j + 1], self.alpha) + R[Li_s, j]
                    )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_aligned < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
