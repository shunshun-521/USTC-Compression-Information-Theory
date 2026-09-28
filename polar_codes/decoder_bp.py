"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（阶段索引因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int) != 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                span = 1 << (n - stage)
                for base in range(0, N, 2 * span):
                    for j in range(span):
                        i = base + j
                        ip = i + span
                        L[i, stage - 1] = self._f_min_sum(
                            R[i, stage - 1] + L[ip, stage], L[i, stage]
                        )
                        L[ip, stage - 1] = self._f_min_sum(
                            R[i, stage - 1], L[i, stage]
                        ) + L[ip, stage]

            for stage in range(0, n):
                span = 1 << (n - stage - 1)
                for base in range(0, N, 2 * span):
                    for j in range(span):
                        i = base + j
                        ip = i + span
                        R[i, stage + 1] = self._f_min_sum(
                            R[ip, stage + 1] + L[ip, stage + 1], R[i, stage]
                        )
                        R[ip, stage + 1] = self._f_min_sum(
                            R[i, stage], L[ip, stage + 1]
                        ) + R[ip, stage]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
