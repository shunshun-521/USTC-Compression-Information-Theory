"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0, i1 = i + k, i + k + s
                        L[i0, j - 1] = self._f_min_sum(R[i0, j] + L[i1, j], L[i0, j])
                        L[i1, j - 1] = self._f_min_sum(R[i0, j], L[i0, j]) + L[i1, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0, i1 = i + k, i + k + s
                        R[i0, j + 1] = self._f_min_sum(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f_min_sum(R[i0, j], L[i0, j + 1]) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            hard_ch = (llr_ch < 0).astype(int)
            hard_natural = np.zeros(N, dtype=int)
            for i in range(N):
                hard_natural[br[i]] = hard_ch[i]
            if np.array_equal(x_hat, hard_natural):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
