"""
极化码 BP（置信传播）译码器
Flooded min-sum BP，因子图列 0 为信道端（与 Permuted SCD 一致）
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器：列 0 信道端，列 n 信源端"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((N, self.n + 1), dtype=np.float64)

    def _update_l_messages(self):
        for s in range(self.n):
            bs = 1 << (s + 1)
            brs = bs // 2
            for j in range(0, self.N, bs):
                for k in range(brs):
                    idx = j + k
                    self.L[idx, s + 1] = _f_min_sum(
                        self.R[idx, s] + self.L[idx + brs, s],
                        self.L[idx, s],
                        self.alpha,
                    )
                    self.L[idx + brs, s + 1] = (
                        _f_min_sum(self.R[idx, s], self.L[idx, s], self.alpha)
                        + self.L[idx + brs, s]
                    )

    def _update_r_messages(self):
        for s in range(self.n - 1, -1, -1):
            bs = 1 << (s + 1)
            brs = bs // 2
            for j in range(0, self.N, bs):
                for k in range(brs):
                    idx = j + k
                    self.R[idx, s] = _f_min_sum(
                        self.R[idx + brs, s + 1] + self.L[idx + brs, s + 1],
                        self.R[idx, s + 1],
                        self.alpha,
                    )
                    self.R[idx + brs, s] = (
                        _f_min_sum(self.R[idx, s + 1], self.L[idx, s + 1], self.alpha)
                        + self.R[idx + brs, s + 1]
                    )

    def _hard_decision(self):
        total = self.L[:, self.n] + self.R[:, self.n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.L.fill(0.0)
        self.R.fill(0.0)
        self.L[:, 0] = llr_ch
        self.R[:, self.n] = 0.0
        self.R[self.frozen_bits, self.n] = self.LARGE

        num_iters = 0
        u_hat = self._hard_decision()

        for it in range(1, self.max_iter + 1):
            self._update_l_messages()
            self._update_r_messages()
            u_hat = self._hard_decision()
            num_iters = it
            if self._check_early_stop(u_hat, llr_ch):
                break

        return u_hat, num_iters
