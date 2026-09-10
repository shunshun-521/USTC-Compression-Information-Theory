"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype == bool:
            self.frozen_idx = np.where(self.frozen_bits)[0]
        else:
            self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._br = bit_reversal_permutation(N)

    def _hard_bits_from_llr(self, llr):
        bits = (llr < 0).astype(np.int8)
        bits[self.frozen_idx] = 0
        return bits

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self._br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _minsum_f(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 2 ** (j)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx2, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = self._hard_bits_from_llr(total_llr)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = self._hard_bits_from_llr(total_llr)
        return u_hat, num_iters
