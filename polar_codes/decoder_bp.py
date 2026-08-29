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
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.ibr = np.argsort(self.br)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.ibr]
        n, N, alpha = self.n, self.N, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                block = 1 << stage
                half = block >> 1
                for base in range(0, N, block):
                    for i in range(base, base + half):
                        j = i + half
                        L[i, stage - 1] = _f_min_sum(
                            R[i, stage] + L[j, stage], L[i, stage], alpha
                        )
                        L[j, stage - 1] = (
                            _f_min_sum(R[i, stage], L[i, stage], alpha) + L[j, stage]
                        )

            for stage in range(0, n):
                block = 1 << (stage + 1)
                half = block >> 1
                for base in range(0, N, block):
                    for i in range(base, base + half):
                        j = i + half
                        R[i, stage + 1] = _f_min_sum(
                            R[j, stage] + L[j, stage + 1], R[i, stage], alpha
                        )
                        R[j, stage + 1] = (
                            _f_min_sum(R[i, stage], L[i, stage + 1], alpha)
                            + R[j, stage]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters
