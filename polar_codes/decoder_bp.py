"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, x, y):
        sign = np.sign(x) * np.sign(y)
        mag = np.minimum(np.abs(x), np.abs(y))
        return self.alpha * sign * mag

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        large = 1e6

        llr = llr_ch[self.br]
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = large

        num_iters = 0
        hard_ch = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for i in range(n - 1, -1, -1):
                add_k = N // (2 ** (n - i))
                for j in range(0, N, 2 * add_k):
                    for k in range(add_k):
                        a = j + k
                        b = j + k + add_k
                        L[a, i] = self._f_min_sum(L[a, i + 1], L[b, i + 1] + R[b, i])
                        L[b, i] = self._f_min_sum(R[a, i], L[a, i + 1]) + L[b, i + 1]

            for i in range(0, n):
                add_k = N // (2 ** (n - i))
                for j in range(0, N, 2 * add_k):
                    for k in range(add_k):
                        a = j + k
                        b = j + k + add_k
                        R[a, i + 1] = self._f_min_sum(R[b, i] + L[b, i + 1], R[a, i])
                        R[b, i + 1] = self._f_min_sum(R[a, i], L[a, i + 1]) + R[b, i]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_ch):
                num_iters = it
                return u_hat, num_iters
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
