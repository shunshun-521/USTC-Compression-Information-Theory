"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。返回 (u_hat, num_iters)。
        llr_ch 为信道码字顺序 LLR（与 polar_encode 输出一致）。
        """
        llr_nat = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        L[u, j - 1] = self._f_min_sum(
                            R[u, j] + L[l, j], L[u, j]
                        )
                        L[l, j - 1] = self._f_min_sum(R[u, j], L[u, j]) + L[l, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        R[u, j + 1] = self._f_min_sum(
                            R[l, j] + L[l, j + 1], R[u, j]
                        )
                        R[l, j + 1] = self._f_min_sum(R[u, j], L[u, j + 1]) + R[l, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
