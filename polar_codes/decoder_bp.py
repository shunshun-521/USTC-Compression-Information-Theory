"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import sc_decode_recursive, f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.n = int(np.log2(N))
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _layered_bp(self, llr):
        n, N = self.n, self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                Li = L[i, j] if j < n else L[i, n]
                Lis = L[i + s, j] if j < n else L[i + s, n]
                L[i, j - 1] = self._f_ms(R[i, j] + Lis, Li)
                L[i + s, j - 1] = self._f_ms(R[i, j], Li) + Lis

        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                Li = L[i, j + 1]
                Lis = L[i + s, j + 1]
                R[i, j + 1] = self._f_ms(R[i + s, j + 1] + Lis, R[i, j])
                R[i + s, j + 1] = self._f_ms(R[i, j], Li) + R[i + s, j + 1]

        return L, R

    def _decide(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_br = llr_ch[br].copy()

        num_iters = self.max_iter
        u_hat = None

        for it in range(1, self.max_iter + 1):
            L, R = self._layered_bp(llr_br)
            u_hat = self._decide(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                return u_hat, num_iters

        # min-sum BP 未收敛时，回退 SC 译码
        u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
        return u_hat, num_iters
