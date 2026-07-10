"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _index_matrix(N):
    x = np.arange(1, N + 1)
    n = int(math.log2(N))
    M = np.zeros((N - 1, n), dtype=np.int32)
    for k in range(n):
        step = 2 ** (k + 1)
        half = 2 ** k
        for i in range(0, N, step):
            if i + half < N:
                M[i:i + half, n - k - 1] = x[i:i + half]
    return M.T[M.T > 0].reshape(n, N // 2).T


def _build_mask_dict(N):
    n = int(math.log2(N))
    mask_dict = _index_matrix(N).T - 1
    return mask_dict[np.flip(np.arange(n))]


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.mask_dict = _build_mask_dict(N)

    def _checknode(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_left(self, L, R):
        n, N = self.n, self.N
        for i in range(n - 1, -1, -1):
            i_back = n - i - 1
            add_k = N // (2 ** (i_back + 1))
            for m in self.mask_dict[i]:
                L[i, m] = self._checknode(
                    L[i + 1, m], L[i + 1, m + add_k] + R[i, m + add_k]
                )
                L[i, m + add_k] = self._checknode(
                    R[i, m], L[i + 1, m]
                ) + L[i + 1, m + add_k]

    def _update_right(self, L, R):
        n, N = self.n, self.N
        for i in range(n):
            i_back = n - i - 1
            add_k = N // (2 ** (i_back + 1))
            for m in self.mask_dict[i]:
                R[i + 1, m] = self._checknode(
                    R[i, m], L[i + 1, m + add_k] + R[i, m + add_k]
                )
                R[i + 1, m + add_k] = self._checknode(
                    R[i, m], L[i + 1, m]
                ) + R[i, m + add_k]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch
        R[0] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
