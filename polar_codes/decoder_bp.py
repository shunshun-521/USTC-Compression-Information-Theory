"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _frozen_mask


def _minsum_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for b in range(s):
                        a = i + b
                        c = a + s
                        La = L[a, j + 1]
                        Lc = L[c, j + 1]
                        Ra = R[a, j]
                        L[a, j] = _minsum_f(Ra + Lc, La, self.alpha)
                        L[c, j] = _minsum_f(Ra, La, self.alpha) + Lc

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for b in range(s):
                        a = i + b
                        c = a + s
                        Rc = R[c, j]
                        Lc = L[c, j + 1]
                        Ra = R[a, j]
                        La = L[a, j + 1]
                        R[a, j + 1] = _minsum_f(Rc + Lc, Ra, self.alpha)
                        R[c, j + 1] = _minsum_f(Ra, La, self.alpha) + Rc

            for i in range(N):
                if self.frozen[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
