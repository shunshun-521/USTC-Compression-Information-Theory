"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_check(a, b, alpha):
    """min-sum 校验节点运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _reorder_llr(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        return llr_ch[self.br]

    def _update_left(self, L, R):
        """从右向左更新 L 消息"""
        n, N = self.n, self.N
        for i in range(n - 1, -1, -1):
            add_k = N // (2 ** (n - i))
            for j in range(0, N, 2 * add_k):
                for k in range(add_k):
                    idx = j + k
                    idx2 = j + k + add_k
                    L[idx, i] = _ms_check(
                        L[idx, i + 1],
                        L[idx2, i + 1] + R[idx2, i],
                        self.alpha,
                    )
                    L[idx2, i] = _ms_check(
                        R[idx, i],
                        L[idx, i + 1],
                        self.alpha,
                    ) + L[idx2, i + 1]

    def _update_right(self, L, R):
        """从左向右更新 R 消息"""
        n, N = self.n, self.N
        for i in range(n):
            add_k = N // (2 ** (n - i))
            for j in range(0, N, 2 * add_k):
                for k in range(add_k):
                    idx = j + k
                    idx2 = j + k + add_k
                    R[idx, i + 1] = _ms_check(
                        R[idx, i],
                        L[idx2, i + 1] + R[idx2, i],
                        self.alpha,
                    )
                    R[idx2, i + 1] = _ms_check(
                        R[idx, i],
                        L[idx, i + 1],
                        self.alpha,
                    ) + R[idx2, i]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr = self._reorder_llr(llr_ch)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)
            num_iters = it

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
