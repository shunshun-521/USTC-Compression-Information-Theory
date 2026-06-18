"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._br = bit_reversal_permutation(N)
        self._inv_br = np.zeros(N, dtype=int)
        self._inv_br[self._br] = np.arange(N)

    def _init_messages(self, llr_ch):
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self._br]
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = 1e6
        return L, R

    def _update_left(self, L, R):
        n, N, alpha = self.n, self.N, self.alpha
        for j in range(n - 1, -1, -1):
            s = 1 << j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx = i + k
                    L[idx, j] = _ms_f(
                        R[idx, j + 1] + L[idx + s, j + 1], L[idx, j + 1], alpha
                    )
                    L[idx + s, j] = (
                        _ms_f(R[idx, j + 1], L[idx, j + 1], alpha) + L[idx + s, j + 1]
                    )

    def _update_right(self, L, R):
        n, N, alpha = self.n, self.N, self.alpha
        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx = i + k
                    R[idx, j + 1] = _ms_f(
                        R[idx + s, j] + L[idx + s, j + 1], R[idx, j], alpha
                    )
                    R[idx + s, j + 1] = (
                        _ms_f(R[idx, j], L[idx, j + 1], alpha) + R[idx + s, j]
                    )

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (np.asarray(llr_ch) < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        L, R = self._init_messages(llr_ch)
        u_hat = self._hard_decision(L, R)
        num_iters = 1

        if self._check_early_stop(u_hat, llr_ch):
            return u_hat, num_iters

        for it in range(2, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)
            u_hat = self._hard_decision(L, R)
            num_iters = it
            if self._check_early_stop(u_hat, llr_ch):
                break

        return u_hat, num_iters
