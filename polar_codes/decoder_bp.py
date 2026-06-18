"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        for i in self.frozen_idx:
            R[i, 0] = self.large
        R[np.setdiff1d(np.arange(N), list(self.frozen_idx)), 0] = 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    i2 = i + step
                    L[i, j - 1] = _f_minsum(
                        R[i, j] + L[i2, j], L[i, j], alpha
                    )
                    L[i2, j - 1] = _f_minsum(R[i, j], L[i, j], alpha) + L[i2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    i2 = i + step
                    R[i, j + 1] = _f_minsum(
                        R[i2, j] + L[i2, j + 1], R[i, j], alpha
                    )
                    R[i2, j + 1] = _f_minsum(R[i, j], L[i2, j + 1], alpha) + R[i2, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if i in self.frozen_idx:
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
            if i in self.frozen_idx:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
