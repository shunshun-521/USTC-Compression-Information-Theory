"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        LARGE = 1e8

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for layer in range(n - 1, -1, -1):
                stride = 1 << layer
                for left in range(0, N, 2 * stride):
                    right = left + stride
                    for i in range(stride):
                        a = left + i
                        b = right + i
                        L[a, layer] = _f_min_sum(
                            R[a, layer + 1] + L[b, layer + 1], L[a, layer + 1], self.alpha
                        )
                        L[b, layer] = (
                            _f_min_sum(R[a, layer + 1], L[a, layer + 1], self.alpha)
                            + L[b, layer + 1]
                        )

            for layer in range(n):
                stride = 1 << layer
                for left in range(0, N, 2 * stride):
                    right = left + stride
                    for i in range(stride):
                        a = left + i
                        b = right + i
                        R[a, layer + 1] = _f_min_sum(
                            R[b, layer] + L[b, layer + 1], R[a, layer], self.alpha
                        )
                        R[b, layer + 1] = (
                            _f_min_sum(R[a, layer], L[b, layer + 1], self.alpha) + R[b, layer]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
