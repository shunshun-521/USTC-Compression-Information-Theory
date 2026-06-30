"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e7

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = np.asarray(llr_ch, dtype=np.float64)
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        L[layer, a] = self._f_min_sum(
                            L[layer + 1, a] + R[layer + 1, b], L[layer + 1, b]
                        )
                        L[layer, b] = (
                            self._f_min_sum(R[layer + 1, a], L[layer + 1, a])
                            + L[layer + 1, b]
                        )

            for layer in range(n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        R[layer + 1, a] = self._f_min_sum(
                            R[layer + 1, b] + L[layer + 1, b], R[layer, a]
                        )
                        R[layer + 1, b] = (
                            self._f_min_sum(R[layer, a], L[layer + 1, a])
                            + R[layer + 1, b]
                        )

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
