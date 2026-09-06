"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _index_matrix(N):
    x = np.arange(1, N + 1)
    n = int(np.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2**k
        for i in range(0, N, step):
            if i + half < N:
                M[i : i + half, n - k - 1] = x[i : i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def _build_mask_dict(N):
    n = int(np.log2(N))
    mask = _index_matrix(N).T.astype(int) - 1
    return mask[np.flip(np.arange(n))]


def _min_sum(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图 min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.clip = 1e6
        self.large = self.clip
        self.mask_dict = _build_mask_dict(N)
        self.br = bit_reversal_permutation(N)

    def _checknode(self, x, y):
        return _min_sum(x, y, self.alpha)

    def _update_left(self, R, L):
        m = self.n
        N = self.N
        mask = self.mask_dict
        for i in range(m - 1, -1, -1):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            idx = mask[i]
            if len(idx) == 0:
                continue
            L[i, idx] = self._checknode(L[i + 1, idx], L[i + 1, idx + add_k] + R[i, idx + add_k])
            L[i, idx + add_k] = self._checknode(R[i, idx], L[i + 1, idx]) + L[i + 1, idx + add_k]
        return np.clip(L, -self.clip, self.clip)

    def _update_right(self, R, L):
        m = self.n
        N = self.N
        mask = self.mask_dict
        for i in range(m):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            idx = mask[i]
            if len(idx) == 0:
                continue
            R[i + 1, idx] = self._checknode(R[i, idx], L[i + 1, idx + add_k] + R[i, idx + add_k])
            R[i + 1, idx + add_k] = self._checknode(R[i, idx], L[i + 1, idx]) + R[i, idx + add_k]
        return np.clip(R, -self.clip, self.clip)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_work = llr_ch[self.br]

        m = self.n
        N = self.N
        frozen = self.frozen_bits.astype(bool)

        R = np.zeros((m + 1, N), dtype=np.float64)
        L = np.zeros((m + 1, N), dtype=np.float64)
        R[0] = 0.0
        R[0, frozen] = self.large
        L[m] = llr_work

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L = self._update_left(R, L)
            R = self._update_right(R, L)
            L[m] = llr_work

            total = L[0] + R[0]
            u_hat = (total < 0).astype(int)
            u_hat[frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
