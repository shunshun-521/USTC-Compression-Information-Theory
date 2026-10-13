"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（flooding schedule + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for block in range(0, N, step * 2):
                    for i in range(block, block + step):
                        j = i + step
                        L[i, layer - 1] = _f_min_sum(
                            R[i, layer] + L[j, layer], L[i, layer], self.alpha
                        )
                        L[j, layer - 1] = _f_min_sum(
                            R[i, layer], L[i, layer], self.alpha
                        ) + L[j, layer]

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, step * 2):
                    for i in range(block, block + step):
                        j = i + step
                        R[i, layer + 1] = _f_min_sum(
                            R[j, layer] + L[j, layer + 1], R[i, layer], self.alpha
                        )
                        R[j, layer + 1] = _f_min_sum(
                            R[i, layer], L[i, layer + 1], self.alpha
                        ) + R[j, layer]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
