"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits)
        self.frozen_bits = fb.astype(bool) if fb.dtype != bool else fb
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, alpha = self.n, self.N, self.alpha
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            # L：从信道列向左传播
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        a = i + t
                        b = i + t + s
                        L[a, j] = _f_minsum(R[a, j + 1] + L[b, j + 1], L[a, j + 1], alpha)
                        L[b, j] = _f_minsum(R[a, j + 1], L[a, j + 1], alpha) + L[b, j + 1]

            # R：从源列向右传播
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        a = i + t
                        b = i + t + s
                        R[a, j] = _f_minsum(R[b, j] + L[b, j], R[a, j - 1], alpha)
                        R[b, j] = _f_minsum(R[a, j - 1], L[a, j], alpha) + R[b, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
