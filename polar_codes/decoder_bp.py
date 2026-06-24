"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]

        L[n][:] = llr_ch
        R[0][:] = 0.0
        R[0][self.frozen_bits] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[j - 1][i] = self._f_min_sum(
                        R[j][i] + L[j][i + s],
                        L[j][i],
                    )
                    L[j - 1][i + s] = self._f_min_sum(
                        R[j][i],
                        L[j][i],
                    ) + L[j][i + s]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[j + 1][i] = self._f_min_sum(
                        R[j + 1][i + s] + L[j + 1][i + s],
                        R[j][i],
                    )
                    R[j + 1][i + s] = self._f_min_sum(
                        R[j][i],
                        L[j + 1][i],
                    ) + R[j + 1][i + s]

            num_iters = it
            total = L[0] + R[0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[0] + R[0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
