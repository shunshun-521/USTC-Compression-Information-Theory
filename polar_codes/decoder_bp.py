"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _frozen_array


class BPDecoder:
    """BP 译码器（min-sum，因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _frozen_array(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_pc = llr_ch[br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_pc
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        def hard_bits():
            total = L[:, 0] + R[:, 0]
            u = (total < 0).astype(int)
            u[self.frozen_bits] = 0
            return u

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右 -> 左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j - 1] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_ms(
                            R[idx, j - 1], L[idx, j]
                        ) + L[idx + s, j]

            # 左 -> 右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx + s, j]

            u_hat = hard_bits()
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = hard_bits()
        return u_hat, num_iters
