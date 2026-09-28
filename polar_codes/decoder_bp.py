"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = -np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Rij = R[i, j - 1] if j > 0 else 0.0
                    Li1 = L[i + s, j]
                    Lj1_i = L[i, j]
                    Lj1_ip = L[i + s, j]
                    L[i, j - 1] = _minsum_f(Rij + Li1, Lj1_i, self.alpha)
                    L[i + s, j - 1] = _minsum_f(Rij, Lj1_i, self.alpha) + Lj1_ip

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R_ip_s_j = R[i + s, j + 1] if j + 1 <= n else 0.0
                    L_ip_s_j1 = L[i + s, j + 1]
                    R_ij_m1 = R[i, j - 1] if j > 0 else 0.0
                    L_ij1 = L[i, j + 1]
                    R[i, j + 1] = _minsum_f(
                        R_ip_s_j + L_ip_s_j1, R_ij_m1, self.alpha
                    )
                    R[i + s, j + 1] = _minsum_f(R_ij_m1, L_ij1, self.alpha) + R_ip_s_j

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch > 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
