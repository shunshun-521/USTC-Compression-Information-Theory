"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, permute_channel_llr


LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = L[0] + R[0]
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        llr_perm = permute_channel_llr(llr_ch)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_perm
        R[0, :] = 0.0
        R[0, self.frozen_idx] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = i + k
                        i2 = i + k + s
                        L[j - 1, i1] = _f_min_sum(
                            R[j, i1] + L[j, i2], L[j, i1], alpha
                        )
                        L[j - 1, i2] = _f_min_sum(R[j, i1], L[j, i1], alpha) + L[j, i2]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i1 = i + k
                        i2 = i + k + s
                        R[j + 1, i1] = _f_min_sum(
                            R[j + 1, i2] + L[j + 1, i2], R[j, i1], alpha
                        )
                        R[j + 1, i2] = _f_min_sum(R[j, i1], L[j + 1, i1], alpha) + R[j + 1, i2]

            num_iters = it

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
