"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha
        large = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        L[idx0, j - 1] = _ms_f(
                            R[idx0, j] + L[idx1, j],
                            L[idx0, j],
                            alpha,
                        )
                        L[idx1, j - 1] = _ms_f(R[idx0, j], L[idx0, j], alpha) + L[idx1, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        R[idx0, j] = _ms_f(
                            R[idx1, j] + L[idx1, j],
                            R[idx0, j - 1],
                            alpha,
                        )
                        R[idx1, j] = _ms_f(R[idx0, j - 1], L[idx0, j], alpha) + R[idx1, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat.astype(int), num_iters
