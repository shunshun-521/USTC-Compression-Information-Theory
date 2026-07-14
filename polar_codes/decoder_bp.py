"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self._br]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i:i + s, j - 1] = _f_min_sum(
                        R[i:i + s, j] + L[i + s:i + 2 * s, j],
                        L[i:i + s, j + 1],
                        self.alpha,
                    )
                    L[i + s:i + 2 * s, j - 1] = _f_min_sum(
                        R[i:i + s, j],
                        L[i:i + s, j + 1],
                        self.alpha,
                    ) + L[i + s:i + 2 * s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i:i + s, j + 1] = _f_min_sum(
                        R[i + s:i + 2 * s, j] + L[i + s:i + 2 * s, j + 1],
                        R[i:i + s, j],
                        self.alpha,
                    )
                    R[i + s:i + 2 * s, j + 1] = _f_min_sum(
                        R[i:i + s, j],
                        L[i:i + s, j + 1],
                        self.alpha,
                    ) + R[i + s:i + 2 * s, j]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0 else 1

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
