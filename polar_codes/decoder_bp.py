"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器（极化码因子图上的 min-sum BP）。
    列 0 为信源端，列 n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7
        self.br = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch[self.br]

        for i in range(N):
            if self.frozen_bits[i]:
                R[0, i] = self.LARGE
            else:
                R[0, i] = 0.0

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1
            # 右 -> 左：更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[j - 1, idx] = self._f(
                            R[j - 1, idx] + L[j, idx], L[j, idx + step]
                        )
                        L[j - 1, idx + step] = self._f(
                            R[j - 1, idx], L[j, idx]
                        ) + L[j, idx + step]

            # 左 -> 右：更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[j, idx + step] = self._f(
                            R[j, idx + step] + L[j, idx + step], R[j - 1, idx]
                        )
                        R[j, idx] = self._f(
                            R[j - 1, idx], L[j, idx]
                        ) + R[j, idx + step]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
