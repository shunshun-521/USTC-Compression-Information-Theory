"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        a = i + t
                        b = i + t + s
                        L[a, j] = _minsum_f(
                            R[a, j + 1] + L[b, j + 1],
                            L[a, j + 1],
                            alpha,
                        )
                        L[b, j] = _minsum_f(R[a, j + 1], L[a, j + 1], alpha) + L[b, j + 1]

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        a = i + t
                        b = i + t + s
                        R[a, j + 1] = _minsum_f(
                            R[b, j + 1] + L[b, j + 1],
                            R[a, j],
                            alpha,
                        )
                        R[b, j + 1] = _minsum_f(R[a, j], L[a, j + 1], alpha) + R[b, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = self._hard_bits_from_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, self.max_iter
