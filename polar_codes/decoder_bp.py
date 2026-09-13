"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        frozen_idx = np.where(self.frozen_bits)[0]
        R[:, 0] = 0.0
        R[frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx0 = i + t
                        idx1 = i + t + step
                        L[idx0, j - 1] = _boxplus_min_sum(
                            R[idx0, j] + L[idx1, j], L[idx0, j], alpha
                        )
                        L[idx1, j - 1] = _boxplus_min_sum(
                            R[idx0, j], L[idx0, j], alpha
                        ) + L[idx1, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx0 = i + t
                        idx1 = i + t + step
                        R[idx0, j + 1] = _boxplus_min_sum(
                            R[idx1, j] + L[idx1, j + 1], R[idx0, j], alpha
                        )
                        R[idx1, j + 1] = _boxplus_min_sum(
                            R[idx0, j], L[idx0, j + 1], alpha
                        ) + R[idx1, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
