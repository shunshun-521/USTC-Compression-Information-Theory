"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.inv_rev = np.empty(N, dtype=int)
        self.inv_rev[self.rev] = np.arange(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, alpha = self.n, self.N, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.rev]
        R[:, 0] = 0.0
        R[self.frozen_bits[self.rev], 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for phi in range(0, N, 2 * step):
                    for beta in range(step):
                        idx = phi + beta
                        L[idx, layer] = _ms_f(
                            L[idx, layer + 1] + R[idx, layer],
                            L[idx + step, layer + 1],
                            alpha,
                        )
                        L[idx + step, layer] = (
                            _ms_f(L[idx, layer + 1], R[idx, layer], alpha) + L[idx + step, layer + 1]
                        )

            for layer in range(n):
                step = 1 << layer
                for phi in range(0, N, 2 * step):
                    for beta in range(step):
                        idx = phi + beta
                        R[idx, layer + 1] = _ms_f(
                            R[idx, layer], R[idx + step, layer] + L[idx + step, layer], alpha
                        )
                        R[idx + step, layer + 1] = (
                            _ms_f(R[idx, layer], L[idx, layer], alpha) + R[idx + step, layer]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[self.inv_rev[i]] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        return u_hat, num_iters
