"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e10
        self._br = bit_reversal_permutation(N)
        self._br_inv = np.argsort(self._br)

    def _f_op(self, a, b):
        a = float(a)
        b = float(b)
        sign = np.sign(a) * np.sign(b)
        if sign == 0:
            sign = 1.0
        return self.alpha * sign * min(abs(a), abs(b))

    def _hard_bits_internal(self, L, R, frozen_internal):
        u_hat = np.zeros(self.N, dtype=np.int32)
        total = L[:, self.n] + R[:, self.n]
        for i in range(self.N):
            if frozen_internal[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def _early_stop(self, u_hat, llr_internal):
        x_hat = polar_encode(u_hat)
        hard = (llr_internal < 0).astype(np.int32)
        return np.array_equal(x_hat[self._br_inv], hard[self._br_inv])

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        llr_internal = llr_ch[self._br]
        frozen_internal = self.frozen_bits[self._br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_internal
        R[:, n] = 0.0
        R[frozen_internal, n] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        L[idx, j + 1] = self._f_op(
                            R[idx, j + 1] + L[idx2, j],
                            L[idx, j],
                        )
                        L[idx2, j + 1] = self._f_op(
                            R[idx, j + 1],
                            L[idx, j],
                        ) + L[idx2, j]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        R[idx, j] = self._f_op(
                            R[idx2, j + 1] + L[idx2, j],
                            R[idx, j + 1],
                        )
                        R[idx2, j] = self._f_op(
                            R[idx, j + 1],
                            L[idx, j],
                        ) + R[idx2, j + 1]

            u_internal = self._hard_bits_internal(L, R, frozen_internal)
            if self._early_stop(u_internal, llr_internal):
                break

        u_internal = self._hard_bits_internal(L, R, frozen_internal)
        u_hat = np.zeros(N, dtype=np.int32)
        u_hat[self._br] = u_internal
        return u_hat, num_iters
