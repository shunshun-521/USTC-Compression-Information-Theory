"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch[self.rev]

        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右 -> 左更新 L（列 n 为信道观测）
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                jc = min(j + 1, n)
                for i in range(0, N, 2 * step):
                    s = step
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + s, jc], L[i, jc]
                    )
                    L[i + s, j - 1] = self._f_min_sum(
                        R[i, j], L[i, jc]
                    ) + L[i + s, jc]

            # 左 -> 右更新 R
            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                jc = min(j + 1, n)
                for i in range(0, N, 2 * step):
                    s = step
                    R[i, j] = self._f_min_sum(
                        R[i + s, j] + L[i + s, jc], R[i, j - 1]
                    )
                    R[i + s, j] = self._f_min_sum(
                        R[i, j - 1], L[i, jc]
                    ) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
