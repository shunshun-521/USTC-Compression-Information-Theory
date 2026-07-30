"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for layer in range(n, 0, -1):
                step = 2 ** (layer - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        L[a, layer - 1] = self._f_min_sum(
                            R[a, layer] + L[b, layer], L[a, layer]
                        )
                        L[b, layer - 1] = self._f_min_sum(
                            R[a, layer], L[a, layer]
                        ) + L[b, layer]

            for layer in range(0, n):
                step = 2 ** layer
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        R[a, layer + 1] = self._f_min_sum(
                            R[b, layer] + L[b, layer + 1], R[a, layer]
                        )
                        R[b, layer + 1] = self._f_min_sum(
                            R[a, layer], L[a, layer + 1]
                        ) + R[b, layer]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if (total >= 0 and not self.frozen_bits[i]) else 0
                if not self.frozen_bits[i]:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
