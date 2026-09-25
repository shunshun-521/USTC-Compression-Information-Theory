"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

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


def _min_sum(y1, y2, alpha=0.9375):
    return alpha * np.sign(y1) * np.sign(y2) * np.minimum(np.abs(y1), np.abs(y2))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.clip = 1e6

        mask_dict = _index_matrix(N).T - 1
        self.mask_dict = mask_dict[np.flip(np.arange(self.n))]

    def _checknode(self, y1, y2):
        return _min_sum(y1, y2, self.alpha)

    def _update_left(self, R, L):
        for i in range(self.n):
            i_back = self.n - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            L[:, i, mask] = self._checknode(
                L[:, i + 1, mask],
                L[:, i + 1, mask + add_k] + R[:, i, mask + add_k],
            )
            L[:, i, mask + add_k] = (
                self._checknode(R[:, i, mask], L[:, i + 1, mask])
                + L[:, i + 1, mask + add_k]
            )
        return np.clip(L, -self.clip, self.clip)

    def _update_right(self, R, L):
        for i in range(self.n):
            i_back = self.n - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            R[:, i + 1, mask] = self._checknode(
                R[:, i, mask], L[:, i + 1, mask + add_k] + R[:, i, mask + add_k]
            )
            R[:, i + 1, mask + add_k] = (
                self._checknode(R[:, i, mask], L[:, i + 1, mask]) + R[:, i, mask + add_k]
            )
        return np.clip(R, -self.clip, self.clip)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_bp = llr_ch[self.br]
        R = np.zeros((1, self.n + 1, self.N), dtype=np.float64)
        L = np.zeros((1, self.n + 1, self.N), dtype=np.float64)
        R[0, 0, self.frozen_idx] = self.clip
        L[0, self.n, :] = llr_bp

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            total = L[0, 0, :] + R[0, 0, :]
            for i in range(self.N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_bp < 0).astype(int)
            if np.array_equal(x_hat[self.br], hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[0, 0, :] + R[0, 0, :]
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
