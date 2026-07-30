"""
极化码 BP（置信传播）译码器
"""
import numpy as np
import math

from encoder import polar_encode
from channel import hard_decision_llr


def bp_f(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a, b = i + k, i + k + s
                        L[a, j - 1] = bp_f(R[a, j] + L[b, j], L[a, j], self.alpha)
                        L[b, j - 1] = bp_f(R[a, j], L[a, j], self.alpha) + L[b, j]

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a, b = i + k, i + k + s
                        R[a, j + 1] = bp_f(R[b, j] + L[b, j + 1], R[a, j], self.alpha)
                        R[b, j + 1] = bp_f(R[a, j], L[a, j + 1], self.alpha) + R[b, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
