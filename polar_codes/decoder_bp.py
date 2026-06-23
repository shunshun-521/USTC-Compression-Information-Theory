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
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for t in range(n - 1, -1, -1):
                s = 1 << t
                for i in range(0, N, 2 * s):
                    L[i, t] = _f_min_sum(
                        L[i, t + 1], L[i + s, t + 1] + R[i + s, t], self.alpha
                    )
                    L[i + s, t] = _f_min_sum(
                        R[i, t], L[i, t + 1], self.alpha
                    ) + L[i + s, t + 1]

            for t in range(0, n):
                s = 1 << t
                for i in range(0, N, 2 * s):
                    R[i, t + 1] = _f_min_sum(
                        R[i, t], L[i + s, t + 1] + R[i + s, t], self.alpha
                    )
                    R[i + s, t + 1] = _f_min_sum(
                        R[i, t], L[i, t + 1], self.alpha
                    ) + R[i + s, t]

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
