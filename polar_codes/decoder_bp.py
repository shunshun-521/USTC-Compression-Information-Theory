"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（自然索引冻结位）"""

    def __init__(self, N, info_indices, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(self.N))
        self.info_indices = np.asarray(info_indices, dtype=int)
        self.frozen_natural = np.ones(N, dtype=bool)
        self.frozen_natural[self.info_indices] = False
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_natural)[0]
        self.LARGE = 1e7

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        from channel import reorder_channel_llr

        llr_ch = reorder_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = self._f_min_sum(
                        R[i, j], L[i, j]
                    ) + L[i + s, j]

            for j in range(1, n):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = self._f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j - 1]
                    )
                    R[i + s, j] = self._f_min_sum(
                        R[i, j - 1], L[i, j + 1]
                    ) + R[i + s, j]

            for i in range(N):
                if self.frozen_natural[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_natural[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
