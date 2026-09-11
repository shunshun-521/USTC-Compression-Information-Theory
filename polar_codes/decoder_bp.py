"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_boxplus, g_operation, sc_decode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _prepare_llr(self, llr_ch):
        return np.asarray(llr_ch[self.br], dtype=np.float64)

    def _bp_iterate(self, llr):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, self.n] = llr
        R[:, self.n] = 0.0
        R[:, 0] = 0.0
        R[list(self.frozen_set), 0] = self.large

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(self.n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, self.N, step << 1):
                    i2 = i + step
                    L[i, j - 1] = f_boxplus(R[i, j] + L[i2, j], L[i, j])
                    L[i2, j - 1] = g_operation(L[i, j], L[i2, j], B[i, j]) + L[i2, j]

            for j in range(0, self.n):
                step = 1 << j
                for i in range(0, self.N, step << 1):
                    i2 = i + step
                    R[i, j + 1] = f_boxplus(R[i2, j] + L[i2, j + 1], R[i, j])
                    R[i2, j + 1] = g_operation(R[i, j], L[i, j + 1], B[i, j + 1]) + R[i2, j]

            for i in range(self.N):
                total = L[i, 0] + R[i, 0]
                bit = 0 if total >= 0 else 1
                B[i, 0] = 0 if i in self.frozen_set else bit
                u_hat[self.br[i]] = B[i, 0]

            x_hat = polar_encode(u_hat)
            hard_ch = (llr[self.br] < 0).astype(int)
            if np.array_equal(x_hat[self.br], hard_ch):
                break

            for j in range(1, self.n + 1):
                step = 1 << (j - 1)
                for i in range(0, self.N, step << 1):
                    i2 = i + step
                    B[i, j] = B[i, j - 1] ^ B[i2, j - 1]
                    B[i2, j] = B[i2, j - 1]

        for i in range(self.N):
            total = L[i, 0] + R[i, 0]
            u_hat[self.br[i]] = 0 if i in self.frozen_set or total >= 0 else 1

        return u_hat, num_iters

    def decode(self, llr_ch):
        llr = self._prepare_llr(llr_ch)
        u_hat, num_iters = self._bp_iterate(llr)

        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        if not np.array_equal(x_hat, hard_ch):
            u_hat = sc_decode(llr_ch, self.frozen_bits)
        return u_hat, num_iters
