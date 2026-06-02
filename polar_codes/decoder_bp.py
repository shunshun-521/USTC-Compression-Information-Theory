"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _channel_to_natural_llr


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6
        self._rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_nat = _channel_to_natural_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    L[i, j - 1] = _minsum(Ri + Lis, Li, alpha)
                    L[i + s, j - 1] = _minsum(Ri, Li, alpha) + Lis

            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j - 1]
                    Ris = R[i + s, j - 1]
                    R[i, j] = _minsum(Ris + Lis, Ri, alpha)
                    R[i + s, j] = _minsum(Ri, Li, alpha) + Ris

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat
