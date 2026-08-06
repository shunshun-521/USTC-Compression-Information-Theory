"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        Rv = R[idx, j - 1]
                        Lvp1 = L[idx, j]
                        Rvs = R[idx + s, j - 1]
                        Lvsp1 = L[idx + s, j]
                        L[idx, j - 1] = _f_minsum(Rv + Lvsp1, Lvp1, self.alpha)
                        L[idx + s, j - 1] = _f_minsum(Rv, Lvp1, self.alpha) + Lvsp1

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        Rvs = R[idx + s, j + 1]
                        Lvs = L[idx + s, j + 1]
                        Rv = R[idx, j - 1] if j > 0 else R[idx, 0]
                        Lvp1 = L[idx, j + 1]
                        R[idx, j + 1] = _f_minsum(Rvs + Lvs, Rv, self.alpha)
                        R[idx + s, j + 1] = _f_minsum(Rv, Lvp1, self.alpha) + Rvs

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            hard_br = np.zeros(N, dtype=int)
            hard_br[br] = hard
            if np.array_equal(x_hat, hard_br):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
