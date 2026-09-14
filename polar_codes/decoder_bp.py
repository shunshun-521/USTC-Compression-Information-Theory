"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, permute_llr_for_decode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N = self.N
        n = self.n
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = permute_llr_for_decode(llr_orig, N)
        hard_ch = (llr_orig < 0).astype(np.int32)

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j - 1] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = (
                            self._f_min_sum(R[idx, j - 1], L[idx, j])
                            + L[idx + s, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = (
                            self._f_min_sum(R[idx, j], L[idx, j + 1])
                            + R[idx + s, j]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            if np.array_equal(polar_encode(u_hat), hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
