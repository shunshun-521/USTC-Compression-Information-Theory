"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.inv_perm = bit_reversal_permutation(N)

    @staticmethod
    def _f_ms(a, b, alpha):
        return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        llr_internal = llr_ch[self.inv_perm]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    L[i, j - 1] = self._f_ms(Ri + Lis, Li, self.alpha)
                    L[i + s, j - 1] = self._f_ms(Ri, Li, self.alpha) + Lis

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j - 1]
                    Ris = R[i + s, j - 1]
                    R[i, j] = self._f_ms(Ris + Lis, Ri, self.alpha)
                    R[i + s, j] = self._f_ms(Ri, Li, self.alpha) + Ris

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

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
