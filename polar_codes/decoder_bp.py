"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（逐层消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        br = bit_reversal_permutation(N)
        self.frozen_decoder_idx = set(
            int(br[i]) for i in np.where(self.frozen_bits == 1)[0]
        )
        self.frozen_natural_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _hard_bits(self, llr_left, llr_right):
        total = llr_left + llr_right
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_natural_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_decoder_idx:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        j2 = i + step
                        L_new[i, j - 1] = _f_min_sum(
                            R[i, j] + L[j2, j], L[i, j], self.alpha
                        )
                        L_new[j2, j - 1] = (
                            _f_min_sum(R[i, j], L[i, j], self.alpha) + L[j2, j]
                        )
            L = L_new

            R_new = R.copy()
            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        j2 = i + step
                        R_new[i, j + 1] = _f_min_sum(
                            R[j2, j + 1] + L[j2, j + 1], R[i, j], self.alpha
                        )
                        R_new[j2, j + 1] = (
                            _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[j2, j]
                        )
            R = R_new

            u_hat = self._hard_bits(L[:, 0], R[:, 0])
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break

        u_hat = self._hard_bits(L[:, 0], R[:, 0])
        return u_hat, num_iters
