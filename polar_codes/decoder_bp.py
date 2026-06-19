"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


def _index_matrix(N):
    x = np.arange(1, N + 1)
    n = int(np.log2(N))
    M = np.zeros((N - 1, n), dtype=int)
    for k in range(n):
        step = 1 << (k + 1)
        half = 1 << k
        for i in range(0, N, step):
            if i + half < N:
                M[i:i + half, n - k - 1] = x[i:i + half]
    valid = M.T[M.T > 0].reshape(n, N // 2).T
    return valid


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        idx = _index_matrix(N)
        self.mask = (idx.T - 1).astype(int)
        self.mask = self.mask[np.flip(np.arange(self.n)), :]

    def _check(self, y1, y2):
        return self.alpha * f_operation(y1, y2)

    def _update_left(self, R, L):
        for i in range(self.n - 1, -1, -1):
            i_back = self.n - i - 1
            add_k = self.N // (1 << (i_back + 1))
            m = self.mask[i]
            L[m, i] = self._check(L[m, i + 1], L[m + add_k, i + 1] + R[m + add_k, i])
            L[m + add_k, i] = self._check(R[m, i], L[m, i + 1]) + L[m + add_k, i + 1]
        return L

    def _update_right(self, R, L):
        for i in range(self.n):
            i_back = self.n - i - 1
            add_k = self.N // (1 << (i_back + 1))
            m = self.mask[i]
            R[m, i + 1] = self._check(R[m, i], L[m + add_k, i + 1] + R[m + add_k, i])
            R[m + add_k, i + 1] = self._check(R[m, i], L[m, i + 1]) + R[m + add_k, i]
        return R

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R[self.frozen_bits, 0] = self.LARGE
        for i in range(self.N):
            L[i, self.n] = llr_ch[self.br[i]]

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
