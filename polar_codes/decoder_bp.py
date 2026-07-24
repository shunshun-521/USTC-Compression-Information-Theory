"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        L[idx0, j - 1] = self._f_min_sum(
                            R[idx0, j] + L[idx1, j],
                            L[idx0, j],
                        )
                        L[idx1, j - 1] = (
                            self._f_min_sum(R[idx0, j], L[idx0, j]) + L[idx1, j]
                        )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        R[idx0, j + 1] = self._f_min_sum(
                            R[idx1, j] + L[idx1, j + 1],
                            R[idx0, j],
                        )
                        R[idx1, j + 1] = (
                            self._f_min_sum(R[idx0, j], L[idx0, j + 1]) + R[idx1, j]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            llr_orig = np.zeros(N)
            llr_orig[self.br] = llr_ch
            x_hard = hard_decision_llr(llr_orig)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
