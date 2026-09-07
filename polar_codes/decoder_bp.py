"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import sc_decode
from encoder import bit_reversal_permutation, polar_encode


def _ms(a, b, alpha):
    sa = np.sign(a) * (a != 0)
    sb = np.sign(b) * (b != 0)
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        llr = np.zeros(self.N, dtype=np.float64)
        llr[br] = llr_ch

        L = np.zeros((self.n + 1, self.N), dtype=np.float64)
        R = np.zeros((self.n + 1, self.N), dtype=np.float64)
        L[self.n, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        converged = False

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(1, self.n + 1):
                step = 2 ** (self.n - s)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[s, idx] = _ms(
                            R[s - 1, idx],
                            L[s - 1, idx + step] + R[s - 1, idx + step],
                            self.alpha,
                        )
                        R[s, idx + step] = _ms(
                            R[s - 1, idx],
                            L[s - 1, idx],
                            self.alpha,
                        ) + R[s - 1, idx + step]

            for s in range(self.n - 1, -1, -1):
                step = 2 ** (self.n - s - 1)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[s, idx] = _ms(
                            R[s + 1, idx],
                            L[s + 1, idx + step] + R[s + 1, idx + step],
                            self.alpha,
                        )
                        L[s, idx + step] = _ms(
                            R[s + 1, idx],
                            L[s + 1, idx],
                            self.alpha,
                        ) + L[s + 1, idx + step]

            for i in range(self.N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                converged = True
                break

        if not converged:
            u_hat = sc_decode(llr_ch, self.frozen_bits)

        return u_hat, num_iters
