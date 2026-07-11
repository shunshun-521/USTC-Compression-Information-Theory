"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（stage 索引：0=信源端，n=信道端）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        best_u = None
        best_iters = self.max_iter

        from encoder import bit_reversal_permutation
        perms = [np.arange(self.N), bit_reversal_permutation(self.N)]

        for perm in perms:
            llr_perm = llr_ch[perm]
            inv = np.argsort(perm)
            u_hat, iters = self._decode_single(llr_perm)
            u_nat = u_hat[inv]
            x_hat = polar_encode(u_nat)
            hard_ch = (llr_ch < 0).astype(np.int64)
            if np.array_equal(x_hat, hard_ch):
                return u_nat, iters
            if best_u is None:
                best_u = u_nat
                best_iters = iters

        return best_u, best_iters

    def _decode_single(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        Z = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        Z[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int64)

        for it in range(self.max_iter):
            num_iters = it + 1

            for s in range(n - 1, -1, -1):
                block = 1 << s
                for t in range(0, N, block << 1):
                    j = t + block
                    Z[t, s] = self._f_min_sum(Z[t, s + 1], Z[j, s + 1] + R[j, s])
                    Z[j, s] = self._f_min_sum(R[t, s], Z[t, s + 1]) + Z[j, s + 1]

            for s in range(0, n):
                block = 1 << s
                for t in range(0, N, block << 1):
                    j = t + block
                    R[t, s + 1] = self._f_min_sum(R[t, s], Z[j, s + 1] + R[j, s])
                    R[j, s + 1] = self._f_min_sum(R[t, s], Z[t, s + 1]) + R[j, s]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (Z[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int64)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (Z[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
