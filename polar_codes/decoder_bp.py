"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _index_matrix(N):
    """极化码因子图各阶段参与更新的索引"""
    x = np.arange(1, N + 1)
    n = int(np.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 1 << (k + 1)
        half = 1 << k
        for i in range(0, N, step):
            if i + half < N:
                M[i:i + half, n - k - 1] = x[i:i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def _min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

        idx_mat = _index_matrix(N).T - 1
        self.mask_dict = [idx_mat[i].astype(int) for i in range(self.n)][::-1]

    def _checknode(self, y1, y2):
        return _min_sum(y1, y2, self.alpha)

    def _update_right(self, R, L, perm):
        m, N = self.n, self.N
        for i in perm:
            i_back = m - i - 1
            add_k = N // (1 << (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            R[i + 1, mask] = self._checknode(
                R[i, mask], L[i + 1, mask + add_k] + R[i, mask + add_k]
            )
            R[i + 1, mask + add_k] = self._checknode(
                R[i, mask], L[i + 1, mask]
            ) + R[i, mask + add_k]
        return R

    def _update_left(self, R, L, perm):
        m, N = self.n, self.N
        for i in perm[::-1]:
            i_back = m - i - 1
            add_k = N // (1 << (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            L[i, mask] = self._checknode(
                L[i + 1, mask], L[i + 1, mask + add_k] + R[i, mask + add_k]
            )
            L[i, mask + add_k] = self._checknode(
                R[i, mask], L[i + 1, mask]
            ) + L[i + 1, mask + add_k]
        return L

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m, N = self.n, self.N

        R = np.zeros((m + 1, N), dtype=np.float64)
        L = np.zeros((m + 1, N), dtype=np.float64)
        R[0, self.frozen_idx] = self.LARGE
        L[m] = llr_ch

        perm = np.arange(m)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            L = self._update_left(R, L, perm)
            R = self._update_right(R, L, perm)
            num_iters = it

            total = L[0] + R[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[0] + R[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
