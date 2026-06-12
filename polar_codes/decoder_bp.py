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
    L[stage, i] / R[stage, i]：stage=0 为信源端，stage=n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        idx1 = i + j
                        idx2 = i + j + step
                        L[stage, idx1] = _f_min_sum(
                            R[stage, idx1] + L[stage + 1, idx2],
                            L[stage + 1, idx1],
                            self.alpha,
                        )
                        L[stage, idx2] = (
                            _f_min_sum(
                                R[stage, idx1],
                                L[stage + 1, idx1],
                                self.alpha,
                            )
                            + L[stage + 1, idx2]
                        )

            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        idx1 = i + j
                        idx2 = i + j + step
                        R[stage + 1, idx1] = _f_min_sum(
                            R[stage, idx2] + L[stage + 1, idx2],
                            R[stage, idx1],
                            self.alpha,
                        )
                        R[stage + 1, idx2] = (
                            _f_min_sum(
                                R[stage, idx1],
                                L[stage + 1, idx2],
                                self.alpha,
                            )
                            + R[stage, idx2]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[0, i] + R[0, i]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_bits_from_llr(llr_ch)):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
