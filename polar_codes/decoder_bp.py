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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(frozen_bits.astype(bool))[0])
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_idx:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    La = R[i, j - 1] + L[i + s, j]
                    Lb = L[i, j]
                    L[i, j - 1] = _minsum_f(La, Lb, self.alpha)
                    Ra = R[i, j - 1]
                    Lb2 = L[i, j]
                    Lc = L[i + s, j]
                    L[i + s, j - 1] = _minsum_f(Ra, Lb2, self.alpha) + Lc

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Rb = R[i + s, j] + L[i + s, j + 1]
                    Ra = R[i, j - 1] if j > 0 else R[i, 0]
                    R[i, j] = _minsum_f(Rb, Ra, self.alpha)
                    Rb2 = R[i, j - 1] if j > 0 else R[i, 0]
                    Lb = L[i, j + 1]
                    Rc = R[i + s, j]
                    R[i + s, j] = _minsum_f(Rb2, Lb, self.alpha) + Rc

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_idx:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if i in self.frozen_idx:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat, num_iters
