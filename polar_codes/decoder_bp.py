"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        idx1 = block + i
                        idx2 = block + i + step
                        L[idx1, stage - 1] = self._f_min_sum(
                            R[idx1, stage] + L[idx2, stage], L[idx1, stage]
                        )
                        L[idx2, stage - 1] = self._f_min_sum(
                            R[idx1, stage], L[idx1, stage]
                        ) + L[idx2, stage]

            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        idx1 = block + i
                        idx2 = block + i + step
                        R[idx1, stage] = self._f_min_sum(
                            R[idx2, stage] + L[idx2, stage], R[idx1, stage - 1]
                        )
                        R[idx2, stage] = self._f_min_sum(
                            R[idx1, stage - 1], L[idx1, stage]
                        ) + R[idx2, stage]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
