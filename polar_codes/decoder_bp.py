"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


def _index_matrix(N):
    """构造极化码因子图索引矩阵（与 Kaira/Arikan 一致）。"""
    x = np.arange(1, N + 1)
    n = int(math.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2 ** k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    mat = M.T[M.T > 0].reshape(n, N // 2).T
    return mat


def _build_mask_dict(N):
    n = int(math.log2(N))
    mask = _index_matrix(N).T - 1
    return mask[np.flip(np.arange(n))].astype(int)


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.mask_dict = _build_mask_dict(N)

    def _checknode(self, y1, y2):
        return self.alpha * f_operation(y1, y2)

    def _update_left(self, R, L):
        m = self.n
        N = self.N
        for i in range(m - 1, -1, -1):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            idx = self.mask_dict[i]
            if len(idx) == 0:
                continue
            L[i, idx] = self._checknode(
                L[i + 1, idx], L[i + 1, idx + add_k] + R[i, idx + add_k]
            )
            L[i, idx + add_k] = (
                self._checknode(R[i, idx], L[i + 1, idx]) + L[i + 1, idx + add_k]
            )
        return L

    def _update_right(self, R, L):
        m = self.n
        N = self.N
        for i in range(m):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            idx = self.mask_dict[i]
            if len(idx) == 0:
                continue
            R[i + 1, idx] = self._checknode(
                R[i, idx], L[i + 1, idx + add_k] + R[i, idx + add_k]
            )
            R[i + 1, idx + add_k] = (
                self._checknode(R[i, idx], L[i + 1, idx]) + R[i, idx + add_k]
            )
        return R

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        R = np.zeros((n + 1, N), dtype=np.float64)
        L = np.zeros((n + 1, N), dtype=np.float64)
        R[0, self.frozen_bits] = self.LARGE
        L[n, :] = llr_ch

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int_)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(np.int_)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int_)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(np.int_)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
