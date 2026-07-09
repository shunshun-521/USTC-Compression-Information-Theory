"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((N, self.n + 1), dtype=np.float64)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.R[:, 0] = 0.0
        self.R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_left()
            self._update_right()
            u_hat = self._hard_decision()
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        return self._hard_decision(), num_iters

    def _update_left(self):
        for j in range(self.n):
            s = 1 << j
            for i in range(0, self.N, 2 * s):
                for k in range(s):
                    idx = i + k
                    self.L[idx, j + 1] = _minsum_f(
                        self.R[idx, j] + self.L[idx + s, j],
                        self.L[idx, j],
                        self.alpha,
                    )
                    self.L[idx + s, j + 1] = _minsum_f(
                        self.R[idx, j],
                        self.L[idx, j],
                        self.alpha,
                    ) + self.L[idx + s, j]

    def _update_right(self):
        for j in range(self.n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, self.N, 2 * s):
                for k in range(s):
                    idx = i + k
                    self.R[idx, j - 1] = _minsum_f(
                        self.R[idx + s, j] + self.L[idx + s, j],
                        self.R[idx, j - 1],
                        self.alpha,
                    )
                    self.R[idx + s, j - 1] = _minsum_f(
                        self.R[idx, j - 1],
                        self.L[idx, j],
                        self.alpha,
                    ) + self.R[idx + s, j]

    def _hard_decision(self):
        total = self.L[:, self.n] + self.R[:, self.n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
