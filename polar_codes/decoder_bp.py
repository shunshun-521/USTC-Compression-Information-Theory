"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _reorder_llr


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = _reorder_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        hard_ch = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    Li = R[i, j - 1] + L[i, j]
                    Li_s = L[i + s, j]
                    L[i, j - 1] = _f_min_sum(Li_s, Li, self.alpha)
                    L[i + s, j - 1] = _f_min_sum(R[i, j - 1], L[i, j], self.alpha) + L[i + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    Ri_s = R[i + s, j] + L[i + s, j + 1]
                    R[i, j + 1] = _f_min_sum(Ri_s, R[i, j], self.alpha)
                    R[i + s, j + 1] = _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[i + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, self.max_iter
