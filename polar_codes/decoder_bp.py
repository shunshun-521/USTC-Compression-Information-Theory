"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e10

    def _hard_decision(self, ld, rd):
        total = ld[0] + rd[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        ld = np.zeros((n + 1, N), dtype=np.float64)
        rd = np.zeros((n + 1, N), dtype=np.float64)
        ld[n] = llr_ch[self.br]
        rd[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for l in range(n - 1, -1, -1):
                stride = 1 << (n - 1 - l)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = a + stride
                        ld[l, a] = _f_min_sum(
                            rd[l + 1, a] + ld[l + 1, b], ld[l + 1, a], alpha)
                        ld[l, b] = _f_min_sum(
                            rd[l + 1, a], ld[l + 1, a], alpha) + ld[l + 1, b]

            for l in range(n):
                stride = 1 << (n - 1 - l)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = a + stride
                        rd[l + 1, a] = _f_min_sum(
                            rd[l + 1, b] + ld[l + 1, b], rd[l, a], alpha)
                        rd[l + 1, b] = _f_min_sum(
                            rd[l, a], ld[l + 1, a], alpha) + rd[l + 1, b]

            u_hat = self._hard_decision(ld, rd)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        u_hat = self._hard_decision(ld, rd)
        return u_hat, num_iters
