"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        br = bit_reversal_permutation(N)
        L[:, 0] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx_u = i + t
                        idx_l = i + t + s
                        L[idx_u, j + 1] = _f_minsum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], self.alpha
                        )
                        L[idx_l, j + 1] = _f_minsum(
                            R[idx_u, j], L[idx_u, j], self.alpha
                        ) + L[idx_l, j]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx_u = i + t
                        idx_l = i + t + s
                        R[idx_u, j] = _f_minsum(
                            R[idx_l, j + 1] + L[idx_l, j + 1], R[idx_u, j + 1], self.alpha
                        )
                        R[idx_l, j] = _f_minsum(
                            R[idx_u, j + 1], L[idx_u, j + 1], self.alpha
                        ) + R[idx_l, j + 1]

            total_llr = L[:, n] + R[:, n]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total_llr = L[:, n] + R[:, n]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
