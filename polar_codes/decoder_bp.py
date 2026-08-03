"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_ms(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        L[idx, j - 1] = _f_ms(
                            R[idx, j - 1] + L[idx + step, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + step, j - 1] = _f_ms(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + L[idx + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        R[idx, j + 1] = _f_ms(
                            R[idx + step, j] + L[idx + step, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + step, j + 1] = (
                            _f_ms(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx + step, j]
                        )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
