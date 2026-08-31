"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（自然序 LLR，与 bit-reversal 编码器匹配）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]
        self.rev = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_natural):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_natural < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_natural[self.rev]

        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = _f_min_sum(
                        R[i, j + 1] + L[i + s, j + 1], L[i, j + 1], alpha
                    )
                    L[i + s, j] = _f_min_sum(R[i, j + 1], L[i, j + 1], alpha) + L[i + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = _f_min_sum(R[i, j], L[i, j + 1], alpha) + R[i + s, j]

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_natural):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
