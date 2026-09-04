"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, _map_channel_llr
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _map_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_v = i + k + step
                        L[idx_u, j - 1] = self._f_min_sum(
                            R[idx_u, j] + L[idx_v, j],
                            L[idx_u, j],
                        )
                        L[idx_v, j - 1] = self._f_min_sum(
                            R[idx_u, j],
                            L[idx_u, j],
                        ) + L[idx_v, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_v = i + k + step
                        R[idx_u, j] = self._f_min_sum(
                            R[idx_v, j] + L[idx_v, j],
                            R[idx_u, j - 1],
                        )
                        R[idx_v, j] = self._f_min_sum(
                            R[idx_u, j - 1],
                            L[idx_u, j],
                        ) + R[idx_v, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            br = bit_reversal_permutation(N)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            hard_ch_br = np.zeros(N, dtype=int)
            hard_ch_br[br] = hard_ch
            if np.array_equal(x_hat, hard_ch_br):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
