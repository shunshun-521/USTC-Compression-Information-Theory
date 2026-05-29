"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        llr_u = llr_ch[self.br]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_u
        frozen_perm = self.frozen_bits[self.br]

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            R[0, :] = 0.0
            R[0, frozen_perm] = self.large
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    La = L[layer + 1, i]
                    Lb = L[layer + 1, i + step]
                    Ra = R[layer, i]
                    L[layer, i] = _minsum_f(Ra + Lb, La, self.alpha)
                    L[layer, i + step] = _minsum_f(Ra, La, self.alpha) + Lb

            for layer in range(1, n + 1):
                step = 1 << (layer - 1)
                for i in range(0, N, 2 * step):
                    Li = L[layer, i]
                    Lj = L[layer, i + step]
                    Ri = R[layer - 1, i]
                    Rj = R[layer - 1, i + step]
                    R[layer, i] = _minsum_f(Rj + Lj, Ri, self.alpha)
                    R[layer, i + step] = _minsum_f(Ri, Li, self.alpha) + Rj

            total = L[0, :] + R[0, :]
            u_perm = np.where(total >= 0, 0, 1).astype(int)
            u_perm[frozen_perm] = 0
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.br] = u_perm
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_perm = np.where(total >= 0, 0, 1).astype(int)
        u_perm[frozen_perm] = 0
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.br] = u_perm
        return u_hat, num_iters
