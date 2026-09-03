"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                block = 2 ** (s + 1)
                half = block // 2
                for start in range(0, N, block):
                    for k in range(half):
                        i = start + k
                        j = start + k + half
                        L[i, s] = _minsum_f(R[i, s] + L[j, s + 1], L[i, s + 1], alpha)
                        L[j, s] = _minsum_f(R[i, s], L[i, s + 1], alpha) + L[j, s + 1]

            for s in range(n):
                block = 2 ** (s + 1)
                half = block // 2
                for start in range(0, N, block):
                    for k in range(half):
                        i = start + k
                        j = start + k + half
                        R[i, s + 1] = _minsum_f(R[j, s] + L[j, s + 1], R[i, s], alpha)
                        R[j, s + 1] = _minsum_f(R[i, s], L[i, s + 1], alpha) + R[j, s]

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
