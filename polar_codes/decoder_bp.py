"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（flooding + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, damping=0.7):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.damping = damping
        self.large = 1e6

    def _logsum(self, x, y):
        if x > y:
            return x + np.log1p(np.exp(y - x))
        return y + np.log1p(np.exp(x - y))

    def _f_boxplus(self, x, y):
        return self._logsum(x + y, 0.0) - self._logsum(x, y)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        brp = bit_reversal_permutation(N)
        llr_work = llr_ch[brp]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_work
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            L_new = L.copy()
            R_new = R.copy()

            # 右到左：更新 L[:, j]，j = n-1 ... 0
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        lu = self._f_ms(R[u, j + 1] + L[l, j + 1], L[u, j + 1])
                        ll = self._f_ms(R[u, j + 1], L[u, j + 1]) + L[l, j + 1]
                        L_new[u, j] = self.damping * lu + (1 - self.damping) * L[u, j]
                        L_new[l, j] = self.damping * ll + (1 - self.damping) * L[l, j]

            # 左到右：更新 R[:, j+1]，j = 0 ... n-1
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        ru = self._f_ms(R[l, j + 1] + L[l, j + 1], R[u, j])
                        rl = self._f_ms(R[u, j], L[u, j + 1]) + R[l, j + 1]
                        R_new[u, j + 1] = self.damping * ru + (1 - self.damping) * R[u, j + 1]
                        R_new[l, j + 1] = self.damping * rl + (1 - self.damping) * R[l, j + 1]

            L, R = L_new, R_new

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
