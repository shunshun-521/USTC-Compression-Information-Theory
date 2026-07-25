"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
  因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._init_offsets()

    def _init_offsets(self):
        self.offsets = [1 << i for i in range(self.n + 1)]

    def _f_minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = self.offsets[j - 1]
                for i in range(0, N, 2 * s):
                    idx = np.arange(i, i + s)
                    idx2 = idx + s
                    L[idx, j - 1] = self._f_minsum(R[idx, j] + L[idx2, j], L[idx, j])
                    L[idx2, j - 1] = self._f_minsum(R[idx, j], L[idx, j]) + L[idx2, j]

            for j in range(0, n):
                s = self.offsets[j]
                for i in range(0, N, 2 * s):
                    idx = np.arange(i, i + s)
                    idx2 = idx + s
                    R[idx, j + 1] = self._f_minsum(
                        R[idx2, j] + L[idx2, j + 1], R[idx, j]
                    )
                    R[idx2, j + 1] = self._f_minsum(R[idx, j], L[idx, j + 1]) + R[idx2, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
