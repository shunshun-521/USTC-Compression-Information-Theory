"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        brp = bit_reversal_permutation(N)
        L[:, n] = llr_ch[brp]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # Right to left: update L[:, j] for j = n-1 .. 0
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        Li = L[idx, j]
                        Li_s = L[idx + s, j]
                        Ri = R[idx, j]
                        Ri_s = R[idx + s, j]
                        L[idx, j - 1] = _f_min_sum(Ri + Li_s, Li, alpha)
                        L[idx + s, j - 1] = _f_min_sum(Ri, Li, alpha) + Li_s

            # Left to right: update R[:, j] for j = 0 .. n-1
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        Li = L[idx, j + 1]
                        Li_s = L[idx + s, j + 1]
                        Rj = R[idx, j] if j > 0 else R[idx, 0]
                        Rj_s = R[idx + s, j] if j > 0 else R[idx + s, 0]
                        if j == 0:
                            Rj = R[idx, 0]
                            Rj_s = R[idx + s, 0]
                        R[idx, j + 1] = _f_min_sum(Rj_s + Li_s, Rj, alpha)
                        R[idx + s, j + 1] = _f_min_sum(Rj, Li, alpha) + Rj_s

            # Hard decision
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
