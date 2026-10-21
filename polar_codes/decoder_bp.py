"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化码因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        rev = bit_reversal_permutation(N)
        llr_work = llr_ch[rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_work
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for dep in range(n - 1, -1, -1):
                stride = 1 << dep
                for i in range(0, N, stride << 1):
                    L[i, dep] = _f_min_sum(
                        R[i, dep] + L[i + stride, dep + 1],
                        L[i, dep + 1],
                        self.alpha,
                    )
                    L[i + stride, dep] = (
                        _f_min_sum(R[i, dep], L[i, dep + 1], self.alpha)
                        + L[i + stride, dep + 1]
                    )

            for dep in range(n):
                stride = 1 << dep
                for i in range(0, N, stride << 1):
                    R[i, dep + 1] = _f_min_sum(
                        R[i + stride, dep] + L[i + stride, dep + 1],
                        R[i, dep],
                        self.alpha,
                    )
                    R[i + stride, dep + 1] = (
                        _f_min_sum(R[i, dep], L[i, dep + 1], self.alpha)
                        + R[i + stride, dep]
                    )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
