"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, damping=0.5):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.damping = damping
        self._large = 1e7

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            R_new = R.copy()
            d = self.damping

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        row_u, row_l = i + k, i + k + s
                        lu = self._f(R[row_u, j] + L[row_l, j], L[row_u, j])
                        ll = self._f(R[row_u, j], L[row_u, j]) + L[row_l, j]
                        L_new[row_u, j - 1] = (1 - d) * L[row_u, j - 1] + d * lu
                        L_new[row_l, j - 1] = (1 - d) * L[row_l, j - 1] + d * ll

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        row_u, row_l = i + k, i + k + s
                        r_left = R[row_u, j - 1] if j > 0 else R[row_u, 0]
                        ru = self._f(R[row_l, j] + L[row_l, j + 1], r_left)
                        rl = self._f(r_left, L[row_u, j + 1]) + R[row_l, j]
                        R_new[row_u, j + 1] = (1 - d) * R[row_u, j + 1] + d * ru
                        R_new[row_l, j + 1] = (1 - d) * R[row_l, j + 1] + d * rl

            L, R = L_new, R_new

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
