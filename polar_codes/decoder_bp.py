"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
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
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                span = 2 ** (j - 1)
                for beta in range(0, N, 2 * span):
                    for omega in range(span):
                        i = beta + omega
                        ip = i + span
                        L[i, j - 1] = _f_min_sum(
                            R[i, j] + L[ip, j], L[i, j], alpha
                        )
                        L[ip, j - 1] = _f_min_sum(
                            R[i, j], L[i, j], alpha
                        ) + L[ip, j]

            for j in range(1, n + 1):
                span = 2 ** (j - 1)
                for beta in range(0, N, 2 * span):
                    for omega in range(span):
                        i = beta + omega
                        ip = i + span
                        R[i, j] = _f_min_sum(
                            R[ip, j] + L[ip, j], R[i, j - 1], alpha
                        )
                        R[ip, j] = _f_min_sum(
                            R[i, j - 1], L[i, j], alpha
                        ) + R[ip, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
