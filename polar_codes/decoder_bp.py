"""
极化码 BP（置信传播）译码器：因子图 min-sum，含早停
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（N 行 × (n+1) 列因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右 -> 左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for b in range(step):
                        i0 = i + b
                        i1 = i + b + step
                        L[i0, j - 1] = _ms_f(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _ms_f(R[i0, j], L[i0, j], self.alpha) + L[i1, j]

            # 左 -> 右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for b in range(step):
                        i0 = i + b
                        i1 = i + b + step
                        R[i0, j] = _ms_f(
                            R[i1, j] + L[i1, j], R[i0, j - 1], self.alpha
                        )
                        R[i1, j] = _ms_f(R[i0, j - 1], L[i0, j], self.alpha) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
