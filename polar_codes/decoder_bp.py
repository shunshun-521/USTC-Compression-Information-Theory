"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        """min-sum f 运算（带缩放因子 alpha）。"""
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.rev]

        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = self._f_min_sum(
                        R[i, j], L[i, j]
                    ) + L[i + s, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = self._f_min_sum(
                        R[i + s, j] + L[i + s, j], R[i, j - 1]
                    )
                    R[i + s, j] = self._f_min_sum(
                        R[i, j - 1], L[i, j]
                    ) + R[i + s, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total_llr < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total_llr < 0] = 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
