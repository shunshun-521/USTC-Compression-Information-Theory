"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.inv_br = np.argsort(bit_reversal_permutation(N))
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)[self.inv_br]

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        L[idx_u, j] = self._f_ms(
                            R[idx_u, j + 1] + L[idx_v, j + 1], L[idx_u, j + 1]
                        )
                        L[idx_v, j] = self._f_ms(
                            R[idx_u, j + 1], L[idx_u, j + 1]
                        ) + L[idx_v, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        R[idx_u, j + 1] = self._f_ms(
                            R[idx_v, j] + L[idx_v, j + 1], R[idx_u, j]
                        )
                        R[idx_v, j + 1] = self._f_ms(
                            R[idx_u, j], L[idx_u, j + 1]
                        ) + R[idx_v, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch[self.br]):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
