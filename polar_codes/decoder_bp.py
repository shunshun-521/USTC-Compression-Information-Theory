"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


def _sign_pm(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    if s.ndim == 0:
        return 1.0 if s == 0 else float(s)
    s[s == 0] = 1
    return s


def _f_minsum(a, b, alpha):
    return alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
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

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j] = _f_minsum(
                            R[idx, j + 1] + L[idx + s, j + 1], L[idx, j + 1], self.alpha
                        )
                        L[idx + s, j] = _f_minsum(
                            R[idx, j + 1], L[idx, j + 1], self.alpha
                        ) + L[idx + s, j + 1]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = _f_minsum(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j], self.alpha
                        )
                        R[idx + s, j + 1] = (
                            _f_minsum(R[idx, j], L[idx, j + 1], self.alpha) + R[idx + s, j]
                        )

            # 早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
