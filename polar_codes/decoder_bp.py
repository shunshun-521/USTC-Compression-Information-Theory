"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, inverse_bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（flooded min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._ibr = inverse_bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self._ibr]
        N, n = self.N, self.n

        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        L[n][:] = llr
        R[0][:] = 0.0
        R[0][self.frozen_bits] = 1e6

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                sp = 1 << (j - 1)
                for i in range(0, N, 2 * sp):
                    La = R[j][i : i + sp] + L[j][i + sp : i + 2 * sp]
                    Lb = L[j][i : i + sp]
                    L[j - 1][i : i + sp] = self._f(La, Lb)
                    L[j - 1][i + sp : i + 2 * sp] = (
                        self._f(R[j][i : i + sp], L[j][i : i + sp])
                        + L[j][i + sp : i + 2 * sp]
                    )

            # 左到右更新 R
            for j in range(1, n + 1):
                sp = 1 << (j - 1)
                for i in range(0, N, 2 * sp):
                    R[j][i : i + sp] = self._f(
                        R[j][i + sp : i + 2 * sp] + L[j][i + sp : i + 2 * sp],
                        R[j - 1][i : i + sp],
                    )
                    R[j][i + sp : i + 2 * sp] = (
                        self._f(R[j - 1][i : i + sp], L[j][i : i + sp])
                        + R[j][i + sp : i + 2 * sp]
                    )

            total = L[0] + R[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0] + R[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
