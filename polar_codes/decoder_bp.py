"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]
        self.frozen_idx = np.where(self.frozen_br == 1)[0]

    def _hard_decision(self, L, R):
        u_internal = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_br[i]:
                u_internal[i] = 0
            else:
                u_internal[i] = 0 if (L[i][0] + R[i][0]) >= 0 else 1
        return u_internal[self.br]

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        hard_ch = (llr_natural < 0).astype(int)

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_natural
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _f_min_sum(R[idx, j - 1], L[idx, j], self.alpha) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[idx, j + 1] = _f_min_sum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _f_min_sum(R[idx, j], L[idx2, j + 1], self.alpha) + R[idx2, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
