"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.where(a >= 0, 1.0, -1.0)
    sb = np.where(b >= 0, 1.0, -1.0)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0=信源端，列 n=信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, alpha = self.N, self.n, self.alpha

        # 边消息：从左到右 R[i,j]，从右到左 L[i,j]
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        for it in range(self.max_iter):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, j],
                        L[i, j],
                        alpha,
                    )
                    L[i + s, j - 1] = (
                        _f_min_sum(R[i, j], L[i, j], alpha) + L[i + s, j]
                    )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = _f_min_sum(
                        R[i + s, j] + L[i + s, j],
                        R[i, j - 1],
                        alpha,
                    )
                    R[i + s, j] = (
                        _f_min_sum(R[i, j - 1], L[i, j], alpha) + R[i + s, j - 1]
                    )

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                return u_hat, it + 1

        return self._hard_decision(L, R), self.max_iter

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat
