"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器（自然序信道 LLR，相位序因子图）。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat（自然序）, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                block_size = 1 << (s + 1)
                branch_size = block_size >> 1
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        top = j + k
                        btm = top + branch_size
                        L[top, s + 1] = _f_minsum(
                            R[top, s] + L[btm, s + 1],
                            L[top, s],
                            self.alpha,
                        )
                        L[btm, s + 1] = _f_minsum(
                            R[top, s], L[top, s], self.alpha
                        ) + L[btm, s]

            for s in range(0, n):
                block_size = 1 << (s + 1)
                branch_size = block_size >> 1
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        top = j + k
                        btm = top + branch_size
                        R[top, s + 1] = _f_minsum(
                            R[btm, s] + L[btm, s + 1],
                            R[top, s],
                            self.alpha,
                        )
                        R[btm, s + 1] = (
                            _f_minsum(R[top, s], L[top, s], self.alpha)
                            + R[btm, s]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, n] + R[i, n]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, n] + R[i, n]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
