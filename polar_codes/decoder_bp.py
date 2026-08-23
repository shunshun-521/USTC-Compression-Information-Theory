"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_orig):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_orig < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_orig[br]

        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    idx = np.arange(i, i + s)
                    idx2 = idx + s
                    L[idx, j - 1] = _f_minsum(
                        R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                    )
                    L[idx2, j - 1] = _f_minsum(
                        R[idx, j], L[idx, j], self.alpha
                    ) + L[idx2, j]

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    idx = np.arange(i, i + s)
                    idx2 = idx + s
                    R[idx, j + 1] = _f_minsum(
                        R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                    )
                    R[idx2, j + 1] = _f_minsum(
                        R[idx, j], L[idx, j + 1], self.alpha
                    ) + R[idx2, j]

            u_hat = self._hard_bits(L, R)
            if self._early_stop(u_hat, llr_orig):
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
