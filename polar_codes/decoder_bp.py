"""
极化码 BP（置信传播）译码器
基于因子图 min-sum 近似，含早停；未收敛时以 SC 结果兜底。
"""
import numpy as np
from decoder_sc import f_operation, permute_llr_for_decode, sc_decode_recursive
from encoder import polar_encode


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _early_stop(self, u_hat, llr_perm):
        x_hat = polar_encode(u_hat)
        hard = (llr_perm < 0).astype(int)
        return np.array_equal(x_hat, hard)

    def _bp_iterate(self, llr_perm):
        N, n = self.N, self.n
        frozen = self.frozen_bits
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[frozen, 0] = self.LARGE
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0, i1 = i + k, i + k + s
                        L[i0, j] = self._f_ms(R[i0, j + 1] + L[i1, j + 1], L[i0, j + 1])
                        L[i1, j] = self._f_ms(R[i0, j + 1], L[i0, j + 1]) + L[i1, j + 1]

            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0, i1 = i + k, i + k + s
                        R[i0, j] = self._f_ms(R[i1, j] + L[i1, j], R[i0, j - 1])
                        R[i1, j] = self._f_ms(R[i0, j - 1], L[i0, j]) + R[i1, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[frozen] = 0
            if self._early_stop(u_hat, llr_perm):
                return u_hat, it
        return u_hat, self.max_iter

    def decode(self, llr_ch):
        llr_perm = permute_llr_for_decode(llr_ch)
        u_hat, iters = self._bp_iterate(llr_perm)
        if not self._early_stop(u_hat, llr_perm):
            u_hat = sc_decode_recursive(llr_perm, self.frozen_bits)
            iters = self.max_iter
        return u_hat, iters
