"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._large = 1e7

    def _g(self, x, y):
        return self.alpha * f_operation(x, y)

    def _iterate(self, L, R, llr_ch):
        n = self.n
        N = self.N
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        for stage in range(n - 1, -1, -1):
            step = 1 << stage
            updated = np.zeros(N, dtype=bool)
            for block in range(0, N, 2 * step):
                top, bot = block, block + step
                L[top, stage] = self._g(L[top, stage + 1], L[bot, stage + 1] + R[bot, stage])
                L[bot, stage] = self._g(R[top, stage], L[top, stage + 1]) + L[bot, stage + 1]
                updated[top] = updated[bot] = True
            for i in range(N):
                if not updated[i]:
                    L[i, stage] = L[i, stage + 1]

        for stage in range(n):
            step = 1 << stage
            updated = np.zeros(N, dtype=bool)
            for block in range(0, N, 2 * step):
                top, bot = block, block + step
                R[top, stage + 1] = self._g(R[top, stage], L[bot, stage + 1] + R[bot, stage])
                R[bot, stage + 1] = self._g(R[top, stage], L[top, stage + 1]) + R[bot, stage + 1]
                updated[top] = updated[bot] = True
            for i in range(N):
                if not updated[i]:
                    R[i, stage + 1] = R[i, stage]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            self._iterate(L, R, llr_ch)
            post = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (post[self.info_idx] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                num_iters = it
                return u_hat, num_iters

        post = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (post[self.info_idx] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
