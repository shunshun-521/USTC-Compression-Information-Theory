"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（适配块级蝶形极化码结构）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, num_iters"""
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[self.frozen_idx, n] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(0, n):
                block = 1 << (s + 1)
                half = block // 2
                for p in range(0, N, block):
                    for k in range(half):
                        j = p + k
                        L[j, s + 1] = _f_min_sum(
                            L[j, s] + R[j, s + 1],
                            L[j + half, s],
                            alpha,
                        )
                        L[j + half, s + 1] = _f_min_sum(
                            R[j, s + 1],
                            L[j, s],
                            alpha,
                        ) + L[j + half, s]

            for s in range(n, 0, -1):
                block = 1 << s
                half = block // 2
                for p in range(0, N, block):
                    for k in range(half):
                        j = p + k
                        R[j, s - 1] = _f_min_sum(
                            R[j + half, s],
                            L[j + half, s],
                            alpha,
                        ) + R[j, s]
                        R[j + half, s - 1] = _f_min_sum(
                            R[j, s],
                            L[j, s],
                            alpha,
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, n] + R[i, n]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, n] + R[i, n]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
