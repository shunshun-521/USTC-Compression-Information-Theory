"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _box_plus(x, y):
    """对数域 box-plus（数值稳定）"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ax = np.abs(x)
    ay = np.abs(y)
    sx = np.sign(x)
    sy = np.sign(y)
    t = np.minimum(ax, ay)
    diff = ax - ay
    z = ax + np.log1p(np.exp(-2.0 * ax)) - np.log1p(np.exp(-np.abs(diff)))
    return sx * sy * z


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        if 2 ** self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self._br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        """min-sum 近似（含缩放因子 alpha）"""
        approx = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        exact = _box_plus(a, b)
        return self.alpha * approx + (1.0 - self.alpha) * exact

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self._br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j] = self._f_ms(
                            R[i + k, j + 1] + L[i + k + s, j + 1],
                            L[i + k, j + 1],
                        )
                        L[i + k + s, j] = (
                            self._f_ms(R[i + k, j + 1], L[i + k, j + 1])
                            + L[i + k + s, j + 1]
                        )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j] = self._f_ms(
                            R[i + k + s, j] + L[i + k + s, j],
                            R[i + k, j - 1],
                        )
                        R[i + k + s, j] = (
                            self._f_ms(R[i + k, j - 1], L[i + k, j])
                            + R[i + k + s, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it + 1
                return u_hat, num_iters

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
