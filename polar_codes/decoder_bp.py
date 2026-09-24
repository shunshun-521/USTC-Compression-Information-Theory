"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        self.L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((self.N, self.n + 1), dtype=np.float64)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.L[:, self.n] = llr_ch
        self.R.fill(0.0)
        self.R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_L()
            self._update_R()

            u_hat = self._hard_decision()
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision()
        return u_hat, num_iters

    def _hard_decision(self):
        total = self.L[:, 0] + self.R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _update_L(self):
        for j in range(self.n, 0, -1):
            step = 1 << (j - 1)
            for i in range(0, self.N, step << 1):
                for k in range(step):
                    idx = i + k
                    s = step
                    self.L[idx, j - 1] = _f_minsum(
                        self.R[idx, j] + self.L[idx + s, j],
                        self.L[idx, j],
                        self.alpha,
                    )
                    self.L[idx + s, j - 1] = _f_minsum(
                        self.R[idx, j],
                        self.L[idx, j],
                        self.alpha,
                    ) + self.L[idx + s, j]

    def _update_R(self):
        for j in range(1, self.n + 1):
            step = 1 << (j - 1)
            for i in range(0, self.N, step << 1):
                for k in range(step):
                    idx = i + k
                    s = step
                    self.R[idx, j] = _f_minsum(
                        self.R[idx + s, j] + self.L[idx + s, j],
                        self.R[idx, j - 1],
                        self.alpha,
                    )
                    self.R[idx + s, j] = (
                        _f_minsum(
                            self.R[idx, j - 1],
                            self.L[idx, j],
                            self.alpha,
                        )
                        + self.R[idx + s, j - 1]
                    )

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard)
