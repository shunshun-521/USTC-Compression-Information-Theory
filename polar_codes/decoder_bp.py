"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_ms(x, y, alpha):
    sa = np.sign(x)
    sb = np.sign(y)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                delta = 1 << stage
                for i in range(0, N, 2 * delta):
                    for j in range(delta):
                        idx1 = i + j
                        idx2 = i + j + delta
                        L[idx1, stage] = _f_ms(
                            R[idx2, stage + 1] + L[idx2, stage + 1],
                            L[idx1, stage + 1],
                            alpha,
                        )
                        L[idx2, stage] = (
                            _f_ms(R[idx1, stage + 1], L[idx1, stage + 1], alpha)
                            + L[idx2, stage + 1]
                        )

            for stage in range(n):
                delta = 1 << stage
                for i in range(0, N, 2 * delta):
                    for j in range(delta):
                        idx1 = i + j
                        idx2 = i + j + delta
                        R[idx1, stage + 1] = _f_ms(
                            R[idx2, stage] + L[idx2, stage + 1],
                            R[idx1, stage],
                            alpha,
                        )
                        R[idx2, stage + 1] = (
                            _f_ms(R[idx1, stage], L[idx2, stage + 1], alpha)
                            + R[idx2, stage]
                        )

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
