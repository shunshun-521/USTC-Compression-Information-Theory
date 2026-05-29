"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停
"""
import numpy as np
import math
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（L/R 分层消息，列 0 为信源端，列 n 为信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e10

    def _f(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # R：从左到右（信源 → 信道）
            for stage in range(n):
                step = 1 << stage
                for base in range(0, N, 2 * step):
                    for j in range(base, base + step):
                        R[j, stage + 1] = self._f(
                            R[j, stage] + R[j + step, stage],
                            L[j + step, stage + 1],
                        )
                        R[j + step, stage + 1] = self._f(
                            R[j, stage], L[j, stage + 1]
                        ) + R[j + step, stage]

            # L：从右到左（信道 → 信源）
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for base in range(0, N, 2 * step):
                    for j in range(base, base + step):
                        L[j, stage] = self._f(
                            L[j, stage + 1],
                            L[j + step, stage + 1] + R[j, stage + 1],
                        )
                        L[j + step, stage] = self._f(
                            L[j, stage + 1], R[j + step, stage + 1]
                        ) + L[j + step, stage + 1]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat.astype(int), num_iters
