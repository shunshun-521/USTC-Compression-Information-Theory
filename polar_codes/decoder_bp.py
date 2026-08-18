"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum(self, x, y):
        return self.alpha * f_operation(x, y)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        br = bit_reversal_permutation(N)
        L[:, n] = llr_ch[br]
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        a = block + i
                        b = a + stride
                        L[a, stage] = self._minsum(
                            R[a, stage + 1] + L[b, stage + 1],
                            L[a, stage + 1],
                        )
                        L[b, stage] = (
                            self._minsum(R[a, stage + 1], L[a, stage + 1]) + L[b, stage + 1]
                        )

            for stage in range(1, n + 1):
                stride = 1 << (stage - 1)
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        a = block + i
                        b = a + stride
                        R[a, stage - 1] = self._minsum(
                            R[b, stage] + L[b, stage],
                            R[a, stage - 1],
                        )
                        R[b, stage - 1] = (
                            self._minsum(R[a, stage - 1], L[a, stage]) + R[b, stage]
                        )

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
