"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _min_sum_f(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    if sa == 0:
        sa = 1.0
    if sb == 0:
        sb = 1.0
    return float(alpha * sa * sb * min(abs(a), abs(b)))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, llr_ch):
        hd = (llr_ch < 0).astype(int)
        hd[self.frozen_bits] = 0
        return hd

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha

        llr = np.zeros((N, n + 1), dtype=np.float64)
        rmsg = np.zeros((N, n + 1), dtype=np.float64)

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr[:, n] = llr_ch
        rmsg[:, 0] = 0.0
        rmsg[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    la = rmsg[i, j] + llr[i + s, j + 1]
                    lb = llr[i, j + 1]
                    llr[i, j] = _min_sum_f(la, lb, alpha)

                    la2 = rmsg[i, j]
                    lb2 = llr[i, j + 1]
                    llr[i + s, j] = _min_sum_f(la2, lb2, alpha) + llr[i + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    la = rmsg[i + s, j + 1] + llr[i + s, j + 1]
                    lb = rmsg[i, j]
                    rmsg[i, j + 1] = _min_sum_f(la, lb, alpha)

                    la2 = rmsg[i, j]
                    lb2 = llr[i, j + 1]
                    rmsg[i + s, j + 1] = _min_sum_f(la2, lb2, alpha) + rmsg[i + s, j]

            total = llr[:, 0] + rmsg[:, 0]
            u_hat[:] = 0
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hd = self._hard_decision(llr_ch)
            if np.array_equal(x_hat, hd):
                num_iters = it
                break

        return u_hat, num_iters
