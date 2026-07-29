"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _bp_f(x, y, alpha):
    """min-sum f 运算，带修正因子 alpha。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        self.inv_brp = np.argsort(bit_reversal_permutation(N))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.inv_brp]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Lj1 = L[i, j]
                    Lj1p = L[i + s, j]
                    L[i, j - 1] = _bp_f(R[i, j] + Lj1p, Lj1, self.alpha)
                    L[i + s, j - 1] = _bp_f(R[i, j], Lj1, self.alpha) + Lj1p

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Rj = R[i, j]
                    Lj1 = L[i, j + 1]
                    R[i, j + 1] = _bp_f(R[i + s, j] + L[i + s, j + 1], Rj, self.alpha)
                    R[i + s, j + 1] = _bp_f(Rj, Lj1, self.alpha) + R[i + s, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
