"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _frozen_bool, reorder_channel_llr


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _frozen_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = reorder_channel_llr(llr_ch)
        N = self.N
        n = self.n
        cols = n + 2
        L = np.zeros((N, cols), dtype=np.float64)
        R = np.zeros((N, cols), dtype=np.float64)

        L[:, n + 1] = llr_ch
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j - 1] = _minsum_f(
                            R[i0, j] + L[i1, j + 1], L[i0, j + 1], self.alpha
                        )
                        L[i1, j - 1] = _minsum_f(
                            R[i0, j], L[i0, j + 1], self.alpha
                        ) + L[i1, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i0, j + 1] = _minsum_f(
                            R[i1, j] + L[i1, j + 1], R[i0, j], self.alpha
                        )
                        R[i1, j + 1] = _minsum_f(
                            R[i0, j], L[i0, j + 1], self.alpha
                        ) + R[i1, j]

            for i in range(N):
                post = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if post >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            post = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if post >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
