"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


def _ms_f(x, y, alpha):
    return alpha * f_operation(x, y)


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        n = self.n
        N = self.N
        llr_ch = llr_ch[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        L[idx_u, j - 1] = _ms_f(
                            R[idx_u, j] + L[idx_v, j], L[idx_u, j], self.alpha
                        )
                        L[idx_v, j - 1] = _ms_f(
                            R[idx_u, j], L[idx_u, j], self.alpha
                        ) + L[idx_v, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        R[idx_u, j + 1] = _ms_f(
                            R[idx_v, j] + L[idx_v, j + 1], R[idx_u, j], self.alpha
                        )
                        R[idx_v, j + 1] = _ms_f(
                            R[idx_u, j], L[idx_u, j + 1], self.alpha
                        ) + R[idx_v, j]

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            x_hat_perm = x_hat[self.br]
            if np.array_equal(x_hat_perm, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            )

        return u_hat, num_iters
