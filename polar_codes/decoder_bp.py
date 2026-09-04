"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import bit_reversal_permutation, _map_channel_llrs

LARGE = 1e8


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_tree = _map_channel_llrs(llr_ch)

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_tree
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        idx2 = idx + s
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j] + L[idx2, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx2, j - 1] = _f_min_sum(
                            R[idx, j],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx2, j]

            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        idx2 = idx + s
                        R[idx, j] = _f_min_sum(
                            R[idx2, j - 1] + L[idx2, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx2, j] = _f_min_sum(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + R[idx2, j - 1]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits.astype(bool)] = 0

        return u_hat, num_iters
