"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self._inv_br = np.argsort(bit_reversal_permutation(N))

    def _hard_bits(self, L, R):
        total = L[0] + R[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _hard_codeword(self, llr_ch):
        return (llr_ch[self._inv_br] < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch.copy()
        R[0] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for layer in range(n, 0, -1):
                sp = 1 << (layer - 1)
                for beta in range(0, N, 2 * sp):
                    for omega in range(sp):
                        i1 = beta + omega
                        i2 = beta + omega + sp
                        L[layer - 1, i1] = _f_min_sum(
                            R[layer - 1, i1] + L[layer, i2],
                            L[layer, i1],
                            self.alpha,
                        )
                        L[layer - 1, i2] = _f_min_sum(
                            R[layer - 1, i1],
                            L[layer, i1],
                            self.alpha,
                        ) + L[layer, i2]

            for layer in range(0, n):
                sp = 1 << layer
                for beta in range(0, N, 2 * sp):
                    for omega in range(sp):
                        i1 = beta + omega
                        i2 = beta + omega + sp
                        R[layer + 1, i1] = _f_min_sum(
                            R[layer + 1, i2] + L[layer + 1, i2],
                            R[layer, i1],
                            self.alpha,
                        )
                        R[layer + 1, i2] = _f_min_sum(
                            R[layer, i1],
                            L[layer + 1, i1],
                            self.alpha,
                        ) + R[layer + 1, i2]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_codeword(llr_ch)):
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
