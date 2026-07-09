"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        br = bit_reversal_permutation(N)
        llr = llr_ch[br].copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i:i + s, j]
                    Lis = L[i + s:i + 2 * s, j]
                    Ri = R[i:i + s, j]
                    L[i:i + s, j - 1] = self._f_min_sum(Ri + Lis, Li)
                    L[i + s:i + 2 * s, j - 1] = self._f_min_sum(Ri, Li) + Lis

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i:i + s, j]
                    Ris = R[i + s:i + 2 * s, j]
                    Lis = L[i + s:i + 2 * s, j + 1]
                    R[i:i + s, j + 1] = self._f_min_sum(Ris + Lis, Ri)
                    R[i + s:i + 2 * s, j + 1] = self._f_min_sum(Ri, Lis) + Ris

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
