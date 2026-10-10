"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 近似；早停与 SC 一致的码字校验。
注：在 min-sum 因子图迭代中采用与 SC 相同的比特倒序与 BRP 信道对齐；
    若迭代未通过早停，回退到 SC 译码以保证链路可用。
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, sc_decode


def _minsum(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _hard_channel_bits(self, llr_ch):
        brp = bit_reversal_permutation(self.N)
        hard = np.zeros(self.N, dtype=int)
        hard[brp] = (llr_ch < 0).astype(int)
        return hard

    def _try_early_stop(self, u_hat, llr_ch):
        return np.array_equal(polar_encode(u_hat), self._hard_channel_bits(llr_ch))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L[:, n] = llr_ch
            R[:, 0] = 0.0
            R[self.frozen_bits, 0] = self.LARGE

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _minsum(R[i, j] + L[i + s, j], L[i, j], self.alpha)
                    L[i + s, j - 1] = _minsum(R[i, j], L[i, j], self.alpha) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _minsum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = _minsum(R[i, j], L[i, j + 1], self.alpha) + R[i + s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            if self._try_early_stop(u_hat, llr_ch):
                return u_hat, num_iters

        u_hat = sc_decode(llr_ch, self.frozen_bits)
        return u_hat, num_iters
