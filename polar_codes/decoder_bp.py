"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br_map = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br_map]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L 消息（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = _f_min_sum(
                        R[i, j] + L[i + s, j + 1], L[i, j + 1], self.alpha
                    )
                    L[i + s, j] = _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + L[i + s, j + 1]

            # 从左到右更新 R 消息（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[i + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
