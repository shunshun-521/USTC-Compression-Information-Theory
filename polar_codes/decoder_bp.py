"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                block = 2 * stride
                for base in range(0, N, block):
                    for k in range(stride):
                        top = base + k
                        bottom = base + k + stride
                        L[top, stage] = self._f_min_sum(
                            L[top, stage + 1],
                            R[bottom, stage] + L[bottom, stage + 1],
                        )
                        L[bottom, stage] = (
                            self._f_min_sum(L[top, stage + 1], R[top, stage])
                            + L[bottom, stage + 1]
                        )

            for stage in range(0, n):
                stride = 1 << stage
                block = 2 * stride
                for base in range(0, N, block):
                    for k in range(stride):
                        top = base + k
                        bottom = base + k + stride
                        R[top, stage + 1] = self._f_min_sum(
                            R[top, stage],
                            R[bottom, stage] + L[bottom, stage + 1],
                        )
                        R[bottom, stage + 1] = (
                            self._f_min_sum(R[top, stage], L[top, stage + 1])
                            + R[bottom, stage]
                        )

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        u_hat[self.frozen_bits] = 0
        for i in range(N):
            if not self.frozen_bits[i]:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
