"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        frozen_idx = np.where(self.frozen_bits.astype(int) != 0)[0]
        self.frozen_idx = frozen_idx
        self.frozen_mask = np.zeros(N, dtype=bool)
        self.frozen_mask[frozen_idx] = True
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    sl = slice(i, i + s)
                    sr = slice(i + s, i + 2 * s)
                    L[sl, j - 1] = self._f_min_sum(
                        R[sl, j] + L[sr, j], L[sl, j]
                    )
                    L[sr, j - 1] = self._f_min_sum(R[sl, j], L[sl, j]) + L[sr, j]

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    sl = slice(i, i + s)
                    sr = slice(i + s, i + 2 * s)
                    R[sl, j + 1] = self._f_min_sum(
                        R[sr, j] + L[sr, j + 1], R[sl, j]
                    )
                    R[sr, j + 1] = self._f_min_sum(R[sl, j], L[sl, j + 1]) + R[sr, j]

            post = L[:, 0] + R[:, 0]
            u_hat = np.where(self.frozen_mask, 0, (post < 0).astype(int))

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                return u_hat, it

            num_iters = it

        return u_hat, num_iters
