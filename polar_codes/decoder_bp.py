"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    sx = np.sign(x)
    sy = np.sign(y)
    sx = np.where(sx == 0, 1.0, sx)
    sy = np.where(sy == 0, 1.0, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = self._hard_bits_from_llr(llr_ch)
        return np.array_equal(x_hat, x_hard)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _ms_f(
                            R[idx, j] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = _ms_f(R[idx, j], L[idx, j], self.alpha) + L[idx + s, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = _ms_f(
                            R[idx + s, j] + L[idx + s, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx + s, j] = _ms_f(R[idx, j - 1], L[idx, j], self.alpha) + R[idx + s, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
