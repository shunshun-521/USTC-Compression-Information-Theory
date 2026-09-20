"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, frozen_mask_to_bool


class BPDecoder:
    """BP 译码器（stage-wise min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = frozen_mask_to_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr = llr_ch[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                span = 1 << (stage - 1)
                for block in range(0, N, span << 1):
                    for i in range(block, block + span):
                        j = i + span
                        L[i, stage - 1] = self._f_ms(
                            R[i, stage] + L[j, stage],
                            L[i, stage],
                        )
                        L[j, stage - 1] = self._f_ms(
                            R[i, stage],
                            L[i, stage],
                        ) + L[j, stage]

            for stage in range(1, n + 1):
                span = 1 << (stage - 1)
                for block in range(0, N, span << 1):
                    for i in range(block, block + span):
                        j = i + span
                        R[i, stage] = self._f_ms(
                            R[j, stage] + L[j, stage],
                            R[i, stage - 1],
                        )
                        R[j, stage] = self._f_ms(
                            R[i, stage - 1],
                            L[i, stage],
                        ) + R[j, stage - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
