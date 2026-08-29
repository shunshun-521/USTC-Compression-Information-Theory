"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（阶段索引：0=信源侧，n=信道侧）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _cn(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_dec = llr_nat[br]

        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_dec
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(n - 1, -1, -1):
                block = 2 ** (stage + 1)
                half = block // 2
                for start in range(0, N, block):
                    for offset in range(half):
                        i = start + offset
                        L[stage, i] = self._cn(
                            L[stage + 1, i],
                            L[stage + 1, i + half] + R[stage, i + half],
                        )
                        L[stage, i + half] = self._cn(
                            R[stage, i],
                            L[stage + 1, i],
                        ) + L[stage + 1, i + half]

            for stage in range(0, n):
                block = 2 ** (stage + 1)
                half = block // 2
                for start in range(0, N, block):
                    for offset in range(half):
                        i = start + offset
                        R[stage + 1, i] = self._cn(
                            R[stage, i],
                            L[stage + 1, i + half] + R[stage, i + half],
                        )
                        R[stage + 1, i + half] = self._cn(
                            R[stage, i],
                            L[stage + 1, i],
                        ) + R[stage, i + half]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hard = (llr_nat < 0).astype(int)
            if np.array_equal(polar_encode(u_hat), x_hard):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
