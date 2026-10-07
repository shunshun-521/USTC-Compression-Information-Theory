"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, _to_tree_frozen


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_natural = np.asarray(frozen_bits, dtype=int)
        self.frozen_tree = _to_tree_frozen(frozen_bits, N)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[np.where(self.frozen_natural == 1)[0], 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # right to left: L messages
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        L[i0, j - 1] = self._f_minsum(
                            R[i0, j] + L[i1, j], L[i0, j]
                        )
                        L[i1, j - 1] = self._f_minsum(R[i0, j], L[i0, j]) + L[i1, j]

            # left to right: R messages
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        R[i0, j + 1] = self._f_minsum(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f_minsum(R[i0, j], L[i0, j + 1]) + R[i1, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        llr_post = L[:, 0] + R[:, 0]
        u_hat = (llr_post < 0).astype(int)
        u_hat[self.frozen_natural.astype(bool)] = 0
        return u_hat.astype(int)
