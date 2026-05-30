"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
LARGE = 1e6


def _sign_pm(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    if s.ndim == 0:
        return 1.0 if s == 0 else float(s)
    s = s.copy()
    s[s == 0] = 1
    return s


def _minsum_f(a, b, alpha=0.9375):
    return alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        BP 译码。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        for it in range(self.max_iter):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j] = _minsum_f(
                            R[idx, j + 1] + L[idx, j + 1],
                            L[idx + s, j + 1],
                            self.alpha,
                        )
                        L[idx + s, j] = (
                            _minsum_f(R[idx, j + 1], L[idx, j + 1], self.alpha)
                            + L[idx + s, j + 1]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = _minsum_f(
                            R[idx + s, j + 1] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = (
                            _minsum_f(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx + s, j + 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it + 1

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
