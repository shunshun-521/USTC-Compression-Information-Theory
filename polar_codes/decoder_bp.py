"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.inf = 1e7

    def _g(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch[self.rev]
        R[0, self.frozen_idx] = self.inf

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        hard_bits = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                d = 1 << s
                for i in range(0, N, 2 * d):
                    for j in range(d):
                        idx = i + j
                        L[s, idx] = self._g(
                            L[s + 1, idx], L[s + 1, idx + d] + R[s, idx + d]
                        )
                        L[s, idx + d] = self._g(L[s + 1, idx], R[s, idx]) + L[s + 1, idx + d]

            for s in range(0, n):
                d = 1 << s
                for i in range(0, N, 2 * d):
                    for j in range(d):
                        idx = i + j
                        R[s + 1, idx] = self._g(
                            R[s, idx], L[s + 1, idx + d] + R[s, idx + d]
                        )
                        R[s + 1, idx + d] = self._g(L[s + 1, idx], R[s, idx]) + R[s, idx + d]

            for i in range(N):
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), hard_bits):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
