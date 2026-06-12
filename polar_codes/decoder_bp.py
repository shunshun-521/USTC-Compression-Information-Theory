"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（按极化因子图阶段更新 L/R 消息）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L = np.zeros((n + 1, N), dtype=np.float64)
            R = np.zeros((n + 1, N), dtype=np.float64)

            L[n] = llr_ch
            R[0] = 0.0
            R[0, self.frozen_bits] = self.large

            for stage in range(n - 1, -1, -1):
                block = 2 ** (stage + 1)
                half = block // 2
                for base in range(0, N, block):
                    for i in range(half):
                        left = base + i
                        right = base + i + half
                        L[stage, left] = self._f(
                            L[stage + 1, left] + R[stage + 1, left],
                            L[stage + 1, right],
                        )
                        L[stage, right] = (
                            self._f(R[stage + 1, left], L[stage + 1, left])
                            + L[stage + 1, right]
                        )

            for stage in range(1, n + 1):
                block = 2 ** stage
                half = block // 2
                for base in range(0, N, block):
                    for i in range(half):
                        left = base + i
                        right = base + i + half
                        R[stage, left] = self._f(
                            R[stage, right] + L[stage, right],
                            R[stage - 1, left],
                        )
                        R[stage, right] = (
                            self._f(R[stage - 1, left], L[stage, left])
                            + R[stage, right]
                        )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
