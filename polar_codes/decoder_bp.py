"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _g_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（flooding schedule，与蝶形+比特倒序编码一致）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e10
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_nat[self.br]
        m = self.m
        N = self.N
        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)
        L[m, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        hard_ch = (llr_nat < 0).astype(int)
        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            Ln = L.copy()
            Rn = R.copy()

            for i in range(m - 1, -1, -1):
                step = 2 ** i
                for j in range(0, N, 2 * step):
                    for k in range(j, j + step):
                        Ln[i, k] = _g_min_sum(
                            L[i + 1, k],
                            L[i + 1, k + step] + R[i, k + step],
                            self.alpha,
                        )
                        Ln[i, k + step] = (
                            _g_min_sum(L[i + 1, k], R[i, k], self.alpha)
                            + L[i + 1, k + step]
                        )

            for i in range(0, m):
                step = 2 ** i
                for j in range(0, N, 2 * step):
                    for k in range(j, j + step):
                        Rn[i + 1, k] = _g_min_sum(
                            R[i, k],
                            L[i + 1, k + step] + R[i, k + step],
                            self.alpha,
                        )
                        Rn[i + 1, k + step] = (
                            _g_min_sum(L[i + 1, k], R[i, k], self.alpha)
                            + R[i, k + step]
                        )

            L, R = Ln, Rn

            for j in range(N):
                if self.frozen_bits[j]:
                    u_hat[j] = 0
                else:
                    u_hat[j] = 0 if (L[0, j] + R[0, j]) >= 0 else 1

            if np.array_equal(polar_encode(u_hat), hard_ch):
                break

        for j in range(N):
            if self.frozen_bits[j]:
                u_hat[j] = 0
            else:
                u_hat[j] = 0 if (L[0, j] + R[0, j]) >= 0 else 1

        return u_hat, num_iters
