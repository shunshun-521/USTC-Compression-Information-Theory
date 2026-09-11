"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _f_sum_product(a, b):
    return float(np.logaddexp(0.0, a + b) - np.logaddexp(a, b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=False):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e10
        self._f = (
            (lambda a, b: _f_min_sum(a, b, alpha))
            if use_min_sum
            else _f_sum_product
        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # stage 0: channel side, stage n: source side
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[0] = llr_ch
        R[n] = 0.0
        R[n, self.frozen_idx] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                span = 1 << stage
                for base in range(0, N, 2 * span):
                    for offset in range(span):
                        i = base + offset
                        j = i + span
                        L[stage + 1, i] = self._f(
                            L[stage, i] + R[stage, j], L[stage, j]
                        )
                        L[stage + 1, j] = self._f(R[stage, i], L[stage, i]) + L[stage, j]

            for stage in range(n - 1, -1, -1):
                span = 1 << stage
                for base in range(0, N, 2 * span):
                    for offset in range(span):
                        i = base + offset
                        j = i + span
                        R[stage, i] = self._f(
                            R[stage + 1, j] + L[stage + 1, j], R[stage + 1, i]
                        )
                        R[stage, j] = (
                            self._f(R[stage + 1, i], L[stage + 1, i]) + R[stage + 1, j]
                        )

            num_iters = it
            total = L[n] + R[n]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[n] + R[n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
