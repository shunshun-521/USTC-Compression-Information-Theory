"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha
        LARGE = 1e6

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[0, i] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(n):
                block = 2 ** (n - stage)
                half = block // 2
                for j in range(0, N, block):
                    for i in range(j, j + half):
                        L[stage, i] = _f_min_sum(
                            R[stage, i] + L[stage + 1, i + half],
                            L[stage + 1, i],
                            alpha,
                        )
                        L[stage, i + half] = (
                            _f_min_sum(R[stage, i], L[stage + 1, i], alpha)
                            + L[stage + 1, i + half]
                        )

            for stage in range(n - 1, -1, -1):
                block = 2 ** (n - stage)
                half = block // 2
                for j in range(0, N, block):
                    for i in range(j, j + half):
                        R[stage + 1, i] = _f_min_sum(
                            R[stage + 1, i + half] + L[stage + 1, i + half],
                            R[stage, i],
                            alpha,
                        )
                        R[stage + 1, i + half] = (
                            _f_min_sum(R[stage, i], L[stage + 1, i], alpha)
                            + R[stage + 1, i + half]
                        )

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
