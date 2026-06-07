"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


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
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _hard_codeword(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                s = 1 << layer
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, layer] = _f_min_sum(
                            R[idx, layer + 1] + L[idx + s, layer + 1],
                            L[idx, layer + 1],
                            self.alpha,
                        )
                        L[idx + s, layer] = _f_min_sum(
                            R[idx, layer],
                            L[idx, layer + 1],
                            self.alpha,
                        ) + L[idx + s, layer + 1]

            for layer in range(n):
                s = 1 << layer
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, layer + 1] = _f_min_sum(
                            R[idx + s, layer + 1] + L[idx + s, layer + 1],
                            R[idx, layer],
                            self.alpha,
                        )
                        R[idx + s, layer + 1] = _f_min_sum(
                            R[idx, layer],
                            L[idx, layer + 1],
                            self.alpha,
                        ) + R[idx + s, layer]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            if np.array_equal(polar_encode(u_hat), self._hard_codeword(llr_ch)):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
