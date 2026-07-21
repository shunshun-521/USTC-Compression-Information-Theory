"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._init_messages()

    def _init_messages(self):
        N, n = self.N, self.n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.R = np.zeros((N, n + 1), dtype=np.float64)

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_L(self):
        N, n = self.N, self.n
        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx1 = i + k
                    idx2 = i + k + s
                    self.L[idx1, j - 1] = self._minsum_f(
                        self.R[idx1, j] + self.L[idx2, j],
                        self.L[idx1, j],
                    )
                    self.L[idx2, j - 1] = self._minsum_f(
                        self.R[idx1, j], self.L[idx1, j]
                    ) + self.L[idx2, j]

    def _update_R(self):
        N, n = self.N, self.n
        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx1 = i + k
                    idx2 = i + k + s
                    self.R[idx1, j + 1] = self._minsum_f(
                        self.R[idx2, j] + self.L[idx2, j + 1],
                        self.R[idx1, j],
                    )
                    self.R[idx2, j + 1] = self._minsum_f(
                        self.R[idx1, j], self.L[idx1, j + 1]
                    ) + self.R[idx2, j]

    def _hard_decision(self):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            total = self.L[i, 0] + self.R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        LARGE = 1e6

        self._init_messages()
        self.L[:, n] = llr_ch
        self.R[:, 0] = 0.0
        self.R[self.frozen_idx, 0] = LARGE

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
