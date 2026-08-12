"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    la = R[i, j - 1] + L[i, j]
                    lb = L[i + s, j]
                    L[i, j - 1] = self._f_min_sum(la, lb)
                    lc = R[i, j - 1]
                    ld = L[i, j]
                    le = L[i + s, j]
                    L[i + s, j - 1] = self._f_min_sum(lc, ld) + le

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    ra = R[i + s, j] + L[i + s, j]
                    rb = R[i, j - 1]
                    R[i, j] = self._f_min_sum(ra, rb)
                    rc = R[i, j - 1]
                    rd = L[i, j]
                    re = R[i + s, j]
                    R[i + s, j] = self._f_min_sum(rc, rd) + re

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                return u_hat, it

        return self._hard_decision(L, R), self.max_iter

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
