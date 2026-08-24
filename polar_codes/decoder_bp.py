"""
极化码 BP（置信传播）译码器
基于因子图蝶形结构，min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（蝶形因子图，scaled min-sum）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def _ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        m = n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(m - 1, -1, -1):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        L[a, j] = self._ms(L[a, j + 1], R[b, j + 1] + L[b, j + 1])
                        L[b, j] = self._ms(L[a, j + 1], R[a, j + 1]) + L[b, j + 1]

            # 从左到右更新 R
            for j in range(0, m):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        R[a, j + 1] = self._ms(R[a, j], R[b, j + 1] + L[b, j + 1])
                        R[b, j + 1] = self._ms(R[a, j], L[a, j + 1]) + R[b, j]

            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if (L[idx, 0] + R[idx, 0]) >= 0 else 1

        return u_hat, num_iters
