"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
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
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        jn = j
                        L[i, j - 1] = _ms_f(
                            R[i, j] + L[i, jn], L[i + step, jn], self.alpha
                        )
                        L[i + step, j - 1] = _ms_f(
                            R[i, j], L[i, jn], self.alpha
                        ) + L[i + step, jn]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        jn = j
                        R[i, j] = _ms_f(
                            R[i + step, j] + L[i + step, jn], R[i, j - 1], self.alpha
                        )
                        R[i + step, j] = _ms_f(
                            R[i, j - 1], L[i, jn], self.alpha
                        ) + R[i + step, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
