"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode

LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for phase in range(n - 1, -1, -1):
                stride = 1 << phase
                for i in range(0, N, 2 * stride):
                    for j in range(i, i + stride):
                        L[j, phase] = _f_min_sum(
                            R[j, phase] + L[j + stride, phase + 1],
                            L[j, phase + 1],
                            self.alpha,
                        )
                        L[j + stride, phase] = (
                            _f_min_sum(R[j, phase], L[j, phase + 1], self.alpha)
                            + L[j + stride, phase + 1]
                        )

            for phase in range(n):
                stride = 1 << phase
                for i in range(0, N, 2 * stride):
                    for j in range(i, i + stride):
                        R[j, phase + 1] = _f_min_sum(
                            R[j + stride, phase] + L[j + stride, phase + 1],
                            R[j, phase],
                            self.alpha,
                        )
                        R[j + stride, phase + 1] = (
                            _f_min_sum(R[j, phase], L[j, phase + 1], self.alpha)
                            + R[j + stride, phase]
                        )

            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
