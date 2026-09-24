"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


def _f_min_sum(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step * 2):
                    L[i, j] = _f_min_sum(
                        R[i, j] + L[i + step, j + 1],
                        L[i, j + 1],
                        self.alpha,
                    )
                    L[i + step, j] = _f_min_sum(
                        R[i, j],
                        L[i, j + 1],
                        self.alpha,
                    ) + L[i + step, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    R[i, j + 1] = _f_min_sum(
                        R[i + step, j] + L[i + step, j + 1],
                        R[i, j],
                        self.alpha,
                    )
                    R[i + step, j + 1] = _f_min_sum(
                        R[i, j],
                        L[i, j + 1],
                        self.alpha,
                    ) + R[i + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
