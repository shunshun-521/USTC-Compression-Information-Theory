"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N = self.N
        n = self.n
        llr = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        j = i + s
                        L[i, stage - 1] = self._minsum(
                            R[i, stage - 1] + L[j, stage],
                            L[i, stage],
                        )
                        L[j, stage - 1] = (
                            self._minsum(R[i, stage - 1], L[i, stage]) + L[j, stage]
                        )

            for stage in range(1, n + 1):
                s = 1 << (stage - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        j = i + s
                        R[i, stage] = self._minsum(
                            R[j, stage] + L[j, stage],
                            R[i, stage - 1],
                        )
                        R[j, stage] = (
                            self._minsum(R[i, stage - 1], L[i, stage]) + R[j, stage]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
