"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停
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
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        for it in range(self.max_iter):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        La = R[j, idx] + L[j, idx + s]
                        Lb = L[j + 1, idx]
                        L[j - 1, idx] = _f_min_sum(La, Lb, alpha)
                        L[j - 1, idx + s] = _f_min_sum(R[j, idx], L[j + 1, idx], alpha) + L[
                            j + 1, idx + s
                        ]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[j, idx] = _f_min_sum(
                            R[j, idx + s] + L[j, idx + s], R[j - 1, idx], alpha
                        )
                        R[j, idx + s] = _f_min_sum(
                            R[j - 1, idx], L[j, idx], alpha
                        ) + R[j, idx + s]

            # 早停
            u_hat = np.zeros(N, dtype=int)
            total = L[0, :] + R[0, :]
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[0, :] + R[0, :]
        u_hat[total >= 0] = 0
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
