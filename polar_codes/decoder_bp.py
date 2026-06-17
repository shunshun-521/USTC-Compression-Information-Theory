"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _permute_channel_llr
from encoder import polar_encode


def _bp_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        llr = _permute_channel_llr(llr_ch, N)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        L_new[a, j - 1] = _bp_f(R[a, j] + L[b, j], L[a, j], self.alpha)
                        L_new[b, j - 1] = _bp_f(R[a, j], L[a, j], self.alpha) + L[b, j]
            L = L_new
            L[:, n] = llr

            R_new = R.copy()
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        R_new[a, j + 1] = _bp_f(R[b, j] + L[b, j + 1], R[a, j], self.alpha)
                        R_new[b, j + 1] = _bp_f(R[a, j], L[b, j + 1], self.alpha) + R[b, j]
            R = R_new
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = self.large

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
