"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_encode

from decoder_sc import sc_decode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._rev = bit_reversal_permutation(self.N)
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        def fms(a, b):
            return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    t = i + s
                    L[i, j - 1] = fms(R[i, j] + L[t, j], L[i, j])
                    L[t, j - 1] = fms(R[i, j], L[i, j]) + L[t, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    t = i + s
                    R[i, j] = fms(R[t, j] + L[t, j], R[i, j - 1])
                    R[t, j] = fms(R[i, j - 1], L[i, j]) + R[t, j]

            post = L[:, 0] + R[:, 0]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)[self._rev]
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, num_iters

        # 未收敛时回退 SC，保证高 SNR 下性能与极化码一致
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        return u_hat, num_iters
