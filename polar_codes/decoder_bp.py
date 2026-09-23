"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr


def _f_min_sum(x, y, alpha):
    sign = np.sign(x) * np.sign(y)
    mag = np.minimum(np.abs(x), np.abs(y))
    return alpha * sign * mag


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self._init_messages()

    def _init_messages(self):
        n, N = self.n, self.N
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.R = np.zeros((N, n + 1), dtype=np.float64)
        self.R[:, 0] = 0.0
        self.R[self.frozen_idx, 0] = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self._init_messages()
        self.L[:, self.n] = llr_ch

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(self.n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, self.N, 2 * s):
                    for k in range(s):
                        i = block + k
                        self.L[i, j - 1] = _f_min_sum(
                            self.R[i, j] + self.L[i + s, j],
                            self.L[i, j],
                            self.alpha,
                        )
                        self.L[i + s, j - 1] = (
                            _f_min_sum(self.R[i, j], self.L[i, j], self.alpha)
                            + self.L[i + s, j]
                        )

            for j in range(1, self.n + 1):
                s = 1 << (j - 1)
                for block in range(0, self.N, 2 * s):
                    for k in range(s):
                        i = block + k
                        self.R[i, j] = _f_min_sum(
                            self.R[i + s, j] + self.L[i + s, j],
                            self.R[i, j - 1],
                            self.alpha,
                        )
                        self.R[i + s, j] = (
                            _f_min_sum(self.R[i, j - 1], self.L[i, j], self.alpha)
                            + self.R[i + s, j]
                        )

            total = self.L[:, 0] + self.R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = self.L[:, 0] + self.R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
