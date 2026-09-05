"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器"""

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
        n = self.n
        N = self.N
        alpha = self.alpha

        self.L.fill(0.0)
        self.R.fill(0.0)
        self.L[:, n] = llr_ch
        self.R[:, 0] = 0.0
        self.R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for iteration in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        ip = i + s
                        self.L[i, stage - 1] = ms_f(
                            self.R[ip, stage - 1] + self.L[ip, stage],
                            self.L[i, stage],
                            alpha,
                        )
                        self.L[ip, stage - 1] = (
                            ms_f(self.R[i, stage - 1], self.L[i, stage], alpha)
                            + self.L[ip, stage]
                        )

            for stage in range(0, n):
                s = 1 << stage
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        ip = i + s
                        self.R[i, stage + 1] = ms_f(
                            self.R[ip, stage] + self.L[ip, stage + 1],
                            self.R[i, stage],
                            alpha,
                        )
                        self.R[ip, stage + 1] = (
                            ms_f(self.R[i, stage], self.L[i, stage + 1], alpha)
                            + self.R[ip, stage]
                        )

            for i in range(N):
                total = self.L[i, 0] + self.R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = iteration
                break
            num_iters = iteration

        for i in range(N):
            total = self.L[i, 0] + self.R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
