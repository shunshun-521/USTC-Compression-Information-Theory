"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int32)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.LARGE

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            R_new = R.copy()

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L_new[i : i + s, j - 1] = self._f_ms(
                        R[i : i + s, j] + L[i + s : i + 2 * s, j],
                        L[i : i + s, j],
                    )
                    L_new[i + s : i + 2 * s, j - 1] = self._f_ms(
                        R[i : i + s, j], L[i : i + s, j]
                    ) + L[i + s : i + 2 * s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    r_prev = R[i : i + s, j - 1] if j > 0 else 0.0
                    R_new[i : i + s, j + 1] = self._f_ms(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                        r_prev,
                    )
                    r_prev2 = R[i : i + s, j - 1] if j > 0 else 0.0
                    R_new[i + s : i + 2 * s, j + 1] = self._f_ms(
                        r_prev2, L[i : i + s, j + 1]
                    ) + R[i + s : i + 2 * s, j]

            L = L_new
            R = R_new
            num_iters = it

            u_hat = np.where(L[:, 0] + R[:, 0] >= 0, 0, 1).astype(np.int32)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = np.where(L[:, 0] + R[:, 0] >= 0, 0, 1).astype(np.int32)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
