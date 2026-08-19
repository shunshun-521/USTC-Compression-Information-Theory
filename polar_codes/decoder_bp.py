"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

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

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        top, bot = i + k, i + k + step
                        L[top, j - 1] = _f_min_sum(
                            R[top, j] + L[bot, j], L[top, j], self.alpha
                        )
                        L[bot, j - 1] = (
                            _f_min_sum(R[top, j], L[top, j], self.alpha)
                            + L[bot, j]
                        )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        top, bot = i + k, i + k + step
                        R[top, j + 1] = _f_min_sum(
                            R[bot, j] + L[bot, j + 1], R[top, j], self.alpha
                        )
                        R[bot, j + 1] = (
                            _f_min_sum(R[top, j], L[top, j + 1], self.alpha)
                            + R[bot, j]
                        )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
