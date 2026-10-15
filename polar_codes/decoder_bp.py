"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _hard_bits(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for layer in range(n - 1, -1, -1):
                stride = 1 << layer
                for base in range(0, N, 2 * stride):
                    for j in range(stride):
                        i = base + j
                        L[i, layer] = _f_min_sum(
                            R[i, layer] + L[i + stride, layer + 1],
                            L[i, layer + 1],
                            self.alpha,
                        )
                        L[i + stride, layer] = _f_min_sum(
                            R[i, layer], L[i, layer + 1], self.alpha
                        ) + L[i + stride, layer + 1]

            for layer in range(0, n):
                stride = 1 << layer
                for base in range(0, N, 2 * stride):
                    for j in range(stride):
                        i = base + j
                        R[i, layer + 1] = _f_min_sum(
                            R[i + stride, layer] + L[i + stride, layer + 1],
                            R[i, layer],
                            self.alpha,
                        )
                        R[i + stride, layer + 1] = _f_min_sum(
                            R[i, layer], L[i + stride, layer + 1], self.alpha
                        ) + R[i + stride, layer]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return self._hard_bits(L, R), num_iters
