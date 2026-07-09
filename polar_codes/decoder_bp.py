"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, step * 2):
                    for i in range(block, block + step):
                        s = step
                        li = i
                        ri = i + s
                        L[i, j - 1] = self._f_min_sum(
                            R[i, j - 1] + L[i + s, j], L[i, j]
                        )
                        L[i + s, j - 1] = self._f_min_sum(
                            R[i, j - 1], L[i, j]
                        ) + L[i + s, j]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, step * 2):
                    for i in range(block, block + step):
                        s = step
                        R[i, j + 1] = self._f_min_sum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j]
                        )
                        R[i + s, j + 1] = self._f_min_sum(
                            R[i, j], L[i, j + 1]
                        ) + R[i + s, j]

            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            x_hat_rev = x_hat[self.rev]
            if np.array_equal(x_hat_rev, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
