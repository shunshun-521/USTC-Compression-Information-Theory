"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum 近似）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        br = bit_reversal_permutation(N)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, step << 1):
                    for j in range(i, i + step):
                        L[j, s] = _boxplus(
                            R[j, s + 1] + L[j + step, s + 1],
                            L[j, s + 1],
                            self.alpha,
                        )
                        L[j + step, s] = (
                            _boxplus(R[j, s + 1], L[j, s + 1], self.alpha)
                            + L[j + step, s + 1]
                        )

            for s in range(n):
                step = 1 << s
                for i in range(0, N, step << 1):
                    for j in range(i, i + step):
                        R[j, s + 1] = _boxplus(
                            R[j + step, s + 1] + L[j + step, s + 1],
                            R[j, s],
                            self.alpha,
                        )
                        R[j + step, s + 1] = (
                            _boxplus(R[j, s], L[j, s + 1], self.alpha)
                            + R[j + step, s]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
