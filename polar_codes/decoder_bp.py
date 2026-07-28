"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（按极化码蝶形因子图迭代更新）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for base in range(0, N, 2 * step):
                    for j in range(step):
                        i = base + j
                        L[i, stage] = self._f(
                            L[i, stage + 1] + R[i, stage], L[i + step, stage + 1]
                        )
                        L[i + step, stage] = (
                            self._f(R[i, stage], L[i, stage + 1]) + L[i + step, stage + 1]
                        )

            for stage in range(n):
                step = 1 << stage
                for base in range(0, N, 2 * step):
                    for j in range(step):
                        i = base + j
                        R[i + step, stage + 1] = (
                            self._f(R[i, stage], L[i + step, stage + 1])
                            + R[i + step, stage]
                        )
                        R[i, stage + 1] = self._f(
                            R[i + step, stage] + L[i + step, stage + 1], R[i, stage]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
