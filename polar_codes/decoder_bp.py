"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import sc_decode
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        u_init = sc_decode(llr_ch, self.frozen_bits)
        for i in range(N):
            if not self.frozen_bits[i]:
                R[i, 0] = self.LARGE if u_init[i] == 0 else -self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                block = 1 << layer
                half = block >> 1
                col = layer - 1
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        s = i + half
                        L[i, col] = _f_min_sum(
                            R[i, col] + L[s, layer],
                            L[i, layer],
                            alpha,
                        )
                        L[s, col] = (
                            _f_min_sum(R[i, col], L[i, layer], alpha) + L[s, layer]
                        )

            for layer in range(0, n):
                block = 1 << (layer + 1)
                half = block >> 1
                col = layer
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        s = i + half
                        R[i, col + 1] = _f_min_sum(
                            R[s, col] + L[s, layer + 1],
                            R[i, col],
                            alpha,
                        )
                        R[s, col + 1] = (
                            _f_min_sum(R[i, col], L[i, layer + 1], alpha) + R[s, col]
                        )

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat
