"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits, dtype=int)
        rev = bit_reversal_permutation(N)
        self.frozen_tree = fb[rev]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.rev = rev

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_tree.astype(bool), 0] = self.large

        num_iters = 0
        u_tree = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    Li, Li2 = i, i + step
                    L[Li, j - 1] = _f_min_sum(
                        R[Li, j] + L[Li2, j], L[Li, j], alpha
                    )
                    L[Li2, j - 1] = _f_min_sum(R[Li, j], L[Li, j], alpha) + L[Li2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Li, Li2 = i, i + step
                    R[Li, j + 1] = _f_min_sum(
                        R[Li2, j] + L[Li2, j + 1], R[Li, j], alpha
                    )
                    R[Li2, j + 1] = _f_min_sum(R[Li, j], L[Li, j + 1], alpha) + R[Li2, j]

            for i in range(N):
                llr_total = L[i, 0] + R[i, 0]
                if self.frozen_tree[i]:
                    u_tree[i] = 0
                else:
                    u_tree[i] = 0 if llr_total >= 0 else 1

            u_hat = u_tree[self.rev]
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = u_tree[self.rev]
        for i in range(N):
            llr_total = L[i, 0] + R[i, 0]
            if self.frozen_tree[i]:
                u_tree[i] = 0
            else:
                u_tree[i] = 0 if llr_total >= 0 else 1
        u_hat = u_tree[self.rev]
        return u_hat, num_iters

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, x_hard)
