"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, inverse_bit_reversal_permutation
from channel import hard_decision_llr


def _index_matrix(N):
    """极化因子图各 stage 的比特索引（参考 Kaira/Arikan BP）"""
    x = np.arange(1, N + 1)
    n = int(math.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2 ** k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def _build_mask_dict(N):
    mat = _index_matrix(N)
    mask = (mat.T - 1).astype(int)
    m = int(math.log2(N))
    return mask[np.arange(m - 1, -1, -1)]


class BPDecoder:
    """BP 译码器（stage-wise min-sum，参考 Kaira BP 结构）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.mask_dict = _build_mask_dict(N)

    def _checknode(self, y1, y2):
        return self.alpha * f_operation(y1, y2)

    def _update_right(self, R, L):
        m = self.n
        N = self.N
        for i in range(m):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            R[i + 1, mask] = self._checknode(
                R[i, mask], L[i + 1, mask + add_k] + R[i, mask + add_k]
            )
            R[i + 1, mask + add_k] = self._checknode(R[i, mask], L[i + 1, mask]) + R[
                i, mask + add_k
            ]
        return np.clip(R, -self.LARGE, self.LARGE)

    def _update_left(self, R, L):
        m = self.n
        N = self.N
        perm = np.arange(m)
        for i in perm[::-1]:
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            L[i, mask] = self._checknode(
                L[i + 1, mask], L[i + 1, mask + add_k] + R[i, mask + add_k]
            )
            L[i, mask + add_k] = self._checknode(R[i, mask], L[i + 1, mask]) + L[
                i + 1, mask + add_k
            ]
        return np.clip(L, -self.LARGE, self.LARGE)

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        inv = inverse_bit_reversal_permutation(self.N)
        llr = llr_raw[inv]
        m = self.n
        N = self.N

        R = np.zeros((m + 1, N), dtype=np.float64)
        L = np.zeros((m + 1, N), dtype=np.float64)
        R[0, self.frozen_bits] = self.LARGE
        L[m, :] = llr

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_raw)):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
