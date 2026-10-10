"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（flooded min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self._br]
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i : i + s, j]
                    Lis = L[i + s : i + 2 * s, j]
                    Ri = R[i : i + s, j]
                    Ris = R[i + s : i + 2 * s, j]
                    L[i : i + s, j - 1] = _f_min_sum(Ri + Lis, Li, self.alpha)
                    L[i + s : i + 2 * s, j - 1] = _f_min_sum(Ri, Li, self.alpha) + Lis

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li = L[i : i + s, j + 1]
                    Lis = L[i + s : i + 2 * s, j + 1]
                    Ri = R[i : i + s, j]
                    Ris = R[i + s : i + 2 * s, j]
                    R[i : i + s, j + 1] = _f_min_sum(Ris + Lis, Ri, self.alpha)
                    R[i + s : i + 2 * s, j + 1] = _f_min_sum(Ri, Li, self.alpha) + Ris

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
        return u_hat.astype(int), num_iters
