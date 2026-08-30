"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _channel_llr_layout, f_operation
from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def _layout_llr(self, llr_ch):
        return _channel_llr_layout(llr_ch)

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = self._layout_llr(llr_ch)
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    Lip1 = L[i, j - 1]
                    Lisp1 = L[i + s, j - 1]

                    L[i, j - 1] = _minsum_f(Ri + Lisp1, Lip1, alpha)
                    L[i + s, j - 1] = _minsum_f(Ri, Lip1, alpha) + Lisp1

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    Lip1 = L[i, j + 1]
                    Lisp1 = L[i + s, j + 1]

                    R[i, j + 1] = _minsum_f(Ris + Lisp1, Ri, alpha)
                    R[i + s, j + 1] = _minsum_f(Ri, Lip1, alpha) + Ris

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
