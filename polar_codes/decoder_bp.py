"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def _update_left(self, L, R):
        n, N = self.n, self.N
        for stage in range(n - 1, -1, -1):
            block = 1 << stage
            step = block << 1
            for start in range(0, N, step):
                for b in range(block):
                    i = start + b
                    j = i + block
                    L[stage, i] = self._f_min_sum(
                        L[stage + 1, i], L[stage + 1, j] + R[stage, j]
                    )
                    L[stage, j] = self._f_min_sum(R[stage, i], L[stage + 1, i]) + L[
                        stage + 1, j
                    ]

    def _update_right(self, L, R):
        n, N = self.n, self.N
        for stage in range(n):
            block = 1 << stage
            step = block << 1
            for start in range(0, N, step):
                for b in range(block):
                    i = start + b
                    j = i + block
                    R[stage + 1, i] = self._f_min_sum(
                        R[stage, i], L[stage + 1, j] + R[stage, j]
                    )
                    R[stage + 1, j] = self._f_min_sum(R[stage, i], L[stage + 1, i]) + R[
                        stage, j
                    ]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = llr_ch[self.br]
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
