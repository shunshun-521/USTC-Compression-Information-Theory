"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（flooded scheduling，列 0 为信源端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        br = bit_reversal_permutation(N)
        llr = llr_ch[br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits[br], 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _minsum_f(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx2, j]

            u_hat_br = self._hard_decision(L, R, br)
            u_hat = np.zeros(N, dtype=int)
            u_hat[br] = u_hat_br
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat_br = self._hard_decision(L, R, br)
        u_hat = np.zeros(N, dtype=int)
        u_hat[br] = u_hat_br
        return u_hat, num_iters

    def _hard_decision(self, L, R, br):
        N = self.N
        frozen_br = self.frozen_bits[br]
        total = L[:, 0] + R[:, 0]
        u_hat_br = (total < 0).astype(int)
        u_hat_br[frozen_br] = 0
        return u_hat_br
