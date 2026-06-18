"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _g_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e7

    def _hard_bits(self, L):
        u_hat = np.zeros(self.N, dtype=int)
        for phi in range(self.N):
            row = self.br[phi]
            u_hat[phi] = 1 if L[row, 0] < 0 else 0
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch[self.br]
        R[:, 0] = 0.0
        for phi in range(N):
            if self.frozen_bits[phi]:
                R[self.br[phi], 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for pj in range(m, 0, -1):
                step = 1 << (pj - 1)
                oj = pj - 1
                op = pj
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        up = i + k
                        lo = up + step
                        L[up, oj] = _g_minsum(L[up, op], L[lo, op] + R[lo, oj], alpha)
                        L[lo, oj] = _g_minsum(R[up, oj], L[up, op], alpha) + L[lo, op]

            for pj in range(1, m + 1):
                step = 1 << (pj - 1)
                oj = pj - 1
                op = pj
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        up = i + k
                        lo = up + step
                        R[up, op] = _g_minsum(R[up, oj], L[lo, op] + R[lo, oj], alpha)
                        R[lo, op] = _g_minsum(R[up, oj], L[up, op], alpha) + R[lo, oj]

            u_hat = self._hard_bits(L)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return self._hard_bits(L), num_iters
