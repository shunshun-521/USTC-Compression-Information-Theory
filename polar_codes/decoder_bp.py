"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode

LARGE = 1e7


class BPDecoder:
    """BP 译码器（按阶段索引的 L/R 消息）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.n
        N = self.N

        # L[i, j]: stage j=1..m+1 (1-indexed in paper), we use 0..m
        # R[i, j]: stage j=0..m
        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # Right to left: update L from stage m down to 1
            for stage in range(m, 0, -1):
                stride = 1 << (stage - 1)
                for block in range(0, N, stride * 2):
                    for node in range(block, block + stride):
                        bot = node + stride
                        L[node, stage - 1] = self._f_ms(
                            L[bot, stage] + R[bot, stage - 1],
                            L[node, stage],
                        )
                        L[bot, stage - 1] = self._f_ms(
                            R[node, stage - 1],
                            L[node, stage],
                        ) + L[bot, stage]

            # Left to right: update R from stage 0 up to m-1
            for stage in range(0, m):
                stride = 1 << stage
                for block in range(0, N, stride * 2):
                    for node in range(block, block + stride):
                        bot = node + stride
                        R[node, stage + 1] = self._f_ms(
                            R[bot, stage] + L[bot, stage + 1],
                            R[node, stage],
                        )
                        R[bot, stage + 1] = self._f_ms(
                            R[node, stage],
                            L[node, stage + 1],
                        ) + R[bot, stage]

            num_iters = it

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
