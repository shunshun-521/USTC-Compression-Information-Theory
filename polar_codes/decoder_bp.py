"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    i2 = i + step
                    L[i : i + step, j - 1] = _ms_f(
                        R[i : i + step, j - 1] + L[i2 : i2 + step, j],
                        L[i : i + step, j],
                        self.alpha,
                    )
                    L[i2 : i2 + step, j - 1] = _ms_f(
                        R[i : i + step, j - 1],
                        L[i : i + step, j],
                        self.alpha,
                    ) + L[i2 : i2 + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    i2 = i + step
                    R[i : i + step, j + 1] = _ms_f(
                        R[i2 : i2 + step, j] + L[i2 : i2 + step, j + 1],
                        R[i : i + step, j],
                        self.alpha,
                    )
                    R[i2 : i2 + step, j + 1] = _ms_f(
                        R[i : i + step, j],
                        L[i : i + step, j + 1],
                        self.alpha,
                    ) + R[i2 : i2 + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            for idx in self.frozen_set:
                u_hat[idx] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        for idx in self.frozen_set:
            u_hat[idx] = 0

        return u_hat, num_iters
