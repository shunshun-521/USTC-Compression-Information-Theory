"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（层索引 0..n，第 n 层为信道 LLR）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 2 ** layer
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        L[layer, a] = _f_min_sum(
                            R[layer + 1, a] + L[layer + 1, b], L[layer + 1, a], self.alpha
                        )
                        L[layer, b] = _f_min_sum(
                            R[layer + 1, a], L[layer + 1, a], self.alpha
                        ) + L[layer + 1, b]

            for layer in range(1, n + 1):
                step = 2 ** (layer - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        R[layer, a] = _f_min_sum(
                            R[layer, b] + L[layer, b], R[layer - 1, a], self.alpha
                        )
                        R[layer, b] = _f_min_sum(
                            R[layer - 1, a], L[layer, a], self.alpha
                        ) + R[layer, b]

            total = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
