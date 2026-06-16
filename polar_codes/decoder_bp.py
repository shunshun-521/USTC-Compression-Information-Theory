"""
极化码 BP（置信传播）译码器
基于因子图（mask 索引更新），含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


def _index_matrix(N):
    x = np.arange(1, N + 1)
    n = int(np.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2 ** k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def _build_mask_dict(N):
    n = int(math.log2(N))
    im = _index_matrix(N)
    return [(im.T - 1)[n - 1 - i] for i in range(n)]


class BPDecoder:
    """BP 译码器。"""

    LLR_MAX = 20.0

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.mask_dict = _build_mask_dict(N)

    def _checknode(self, y1, y2):
        y1 = np.asarray(y1, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)
        out = np.empty_like(y1)
        z1 = np.abs(y1) < 1e-12
        z2 = np.abs(y2) < 1e-12
        out[z1] = y2[z1]
        out[z2 & ~z1] = y1[z2 & ~z1]
        both = ~z1 & ~z2
        out[both] = self.alpha * f_operation(y1[both], y2[both])
        return out

    def _clip(self, arr):
        return np.clip(arr, -self.LLR_MAX, self.LLR_MAX)

    def _hard_codeword(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _update_left(self, R, L):
        for i in range(self.n):
            i_back = self.n - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            L[mask, i] = self._checknode(
                L[mask, i + 1], L[mask + add_k, i + 1] + R[mask + add_k, i]
            )
            L[mask + add_k, i] = (
                self._checknode(R[mask, i], L[mask, i + 1]) + L[mask + add_k, i + 1]
            )
        return self._clip(L)

    def _update_right(self, R, L):
        for i in range(self.n):
            i_back = self.n - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            R[mask, i + 1] = self._checknode(
                R[mask, i], L[mask + add_k, i + 1] + R[mask + add_k, i]
            )
            R[mask + add_k, i + 1] = (
                self._checknode(R[mask, i], L[mask, i + 1]) + R[mask + add_k, i]
            )
        return self._clip(R)

    def _make_decision(self, L, R):
        total = self._clip(L[:, 0] + R[:, 0])
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_perm = self._clip(np.asarray(llr_ch, dtype=np.float64)[self.br])

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LLR_MAX

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L[:, self.n] = llr_perm
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            u_hat = self._make_decision(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_codeword(llr_ch)):
                num_iters = it
                break

        u_hat = self._make_decision(L, R)
        return u_hat, num_iters
