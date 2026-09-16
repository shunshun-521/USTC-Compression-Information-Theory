"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import bit_reversal_permutation
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def _update_left(self, L, R):
        m = self.n
        N = self.N
        for i in range(m):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            for block in range(0, N, 2 * add_k):
                for j in range(block, block + add_k):
                    L[j, i] = _f_min_sum(
                        L[j, i + 1], L[j + add_k, i + 1] + R[j + add_k, i], self.alpha
                    )
                    L[j + add_k, i] = _f_min_sum(R[j, i], L[j, i + 1], self.alpha) + L[j + add_k, i + 1]

    def _update_right(self, L, R):
        m = self.n
        N = self.N
        for i in range(m):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            for block in range(0, N, 2 * add_k):
                for j in range(block, block + add_k):
                    R[j, i + 1] = _f_min_sum(
                        R[j, i], L[j + add_k, i + 1] + R[j + add_k, i], self.alpha
                    )
                    R[j + add_k, i + 1] = _f_min_sum(R[j, i], L[j, i + 1], self.alpha) + R[j + add_k, i]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.n
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch[self.rev]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
