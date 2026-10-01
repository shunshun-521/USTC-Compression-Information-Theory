"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    i2 = i + s
                    Li = L[i : i + s, j]
                    Li2 = L[i2 : i2 + s, j]
                    Ri = R[i : i + s, j]
                    Ri2 = R[i2 : i2 + s, j]
                    L[i : i + s, j - 1] = self._f_ms(Ri + Li2, Li)
                    L[i2 : i2 + s, j - 1] = self._f_ms(Ri, Li) + Li2

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    i2 = i + s
                    Li = L[i : i + s, j + 1]
                    Li2 = L[i2 : i2 + s, j + 1]
                    Ri = R[i : i + s, j]
                    Ri2 = R[i2 : i2 + s, j]
                    R[i : i + s, j + 1] = self._f_ms(Ri2 + Li2, Ri)
                    R[i2 : i2 + s, j + 1] = self._f_ms(Ri, Li) + Ri2

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
