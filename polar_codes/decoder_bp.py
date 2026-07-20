"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        sa, sb = np.sign(a), np.sign(b)
        return self.alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_indices] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            for layer in range(n - 1, -1, -1):
                stride = 1 << layer
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = i + j + stride
                        L[layer, a] = self._f_min_sum(
                            R[layer, a] + L[layer + 1, b],
                            L[layer + 1, a]
                        )
                        L[layer, b] = self._f_min_sum(
                            R[layer, a],
                            L[layer + 1, a]
                        ) + L[layer + 1, b]

            for layer in range(n):
                stride = 1 << layer
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = i + j + stride
                        R[layer + 1, b] = self._f_min_sum(
                            R[layer + 1, a] + L[layer + 1, b],
                            R[layer, a]
                        )
                        R[layer + 1, a] = self._f_min_sum(
                            R[layer, a],
                            L[layer + 1, a]
                        ) + R[layer + 1, b]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
