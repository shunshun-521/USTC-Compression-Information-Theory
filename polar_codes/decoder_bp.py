"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        # L[i, s]: 从右向左消息，s=0 为信源端，s=n 为信道端
        # R[i, s]: 从左向右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        L[j, stage - 1] = self._minsum(
                            R[j, stage] + L[j + step, stage],
                            L[j, stage],
                        )
                        L[j + step, stage - 1] = self._minsum(
                            R[j, stage],
                            L[j, stage],
                        ) + L[j + step, stage]

            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        R[j, stage] = self._minsum(
                            R[j + step, stage] + L[j + step, stage],
                            R[j, stage - 1],
                        )
                        R[j + step, stage] = self._minsum(
                            R[j, stage - 1],
                            L[j, stage],
                        ) + R[j + step, stage]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
