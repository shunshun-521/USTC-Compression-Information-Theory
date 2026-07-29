"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N = self.N
        n = self.n
        alpha = self.alpha
        rev = self.rev

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        llr_internal = np.asarray(llr_ch, dtype=np.float64)[rev]
        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits[rev], 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j], alpha
                    )
                    L[i + s, j - 1] = _f_min_sum(
                        R[i, j], L[i, j], alpha
                    ) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = _f_min_sum(
                        R[i, j], L[i, j + 1], alpha
                    ) + R[i + s, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break
        else:
            u_hat = self._hard_decision(L, R)

        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_internal = (total < 0).astype(int)
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.rev] = u_internal
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (np.asarray(llr_ch) < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
