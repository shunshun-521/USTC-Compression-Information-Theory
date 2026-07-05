"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N, n = self.N, self.n
        ch_llr = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = ch_llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx + s, j + 1], L[idx, j + 1], self.alpha
                        )
                        L[idx + s, j - 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + L[idx + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _minsum_f(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j], self.alpha
                        )
                        R[idx + s, j + 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx + s, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (ch_llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
