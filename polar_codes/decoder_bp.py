"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
"""
import numpy as np
import math

from encoder import polar_encode


def _f_ms(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j - 1] = _f_ms(
                            R[i + k, j] + L[i + k + s, j],
                            L[i + k, j],
                            self.alpha,
                        )
                        L[i + k + s, j - 1] = _f_ms(
                            R[i + k, j],
                            L[i + k, j],
                            self.alpha,
                        ) + L[i + k + s, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j] = _f_ms(
                            R[i + k + s, j] + L[i + k + s, j],
                            R[i + k, j - 1],
                            self.alpha,
                        )
                        R[i + k + s, j] = _f_ms(
                            R[i + k, j - 1],
                            L[i + k, j],
                            self.alpha,
                        ) + R[i + k + s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        return u_hat.astype(int), num_iters
