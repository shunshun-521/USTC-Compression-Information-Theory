"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum / box-plus 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _logsum(a, b):
    if np.isinf(a):
        return b
    if np.isinf(b):
        return a
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + np.log1p(np.exp(-abs(a - b)))


def _boxplus(a, b):
    return _logsum(a + b, 0.0) - _logsum(a, b)


def _g_soft(La, Lb, R_bit):
    return _logsum(La + Lb + R_bit, La - Lb - R_bit) - _logsum(R_bit, -R_bit)


def _ms_f(x, y, alpha):
    sx = np.where(x >= 0, 1.0, -1.0)
    sy = np.where(y >= 0, 1.0, -1.0)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e8

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    L[i, j] = _boxplus(
                        R[i, j] + L[i + s, j + 1], L[i, j + 1]
                    )
                    L[i + s, j] = _g_soft(
                        L[i, j + 1], L[i + s, j + 1], R[i, j]
                    )

            for j in range(n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _boxplus(
                        R[i, j] + L[i + s, j + 1], R[i + s, j + 1]
                    )
                    R[i + s, j + 1] = _boxplus(
                        R[i, j], L[i, j + 1]
                    ) + R[i + s, j + 1]

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
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
