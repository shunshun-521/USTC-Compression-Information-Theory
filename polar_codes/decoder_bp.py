"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _ms_f(
                        R[i, j] + L[i + s, j], L[i, j], self.alpha
                    )
                    L[i + s, j - 1] = (
                        _ms_f(R[i, j], L[i, j], self.alpha) + L[i + s, j]
                    )

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = _ms_f(
                        R[i + s, j] + L[i + s, j], R[i, j - 1], self.alpha
                    )
                    R[i + s, j] = (
                        _ms_f(R[i, j - 1], L[i, j], self.alpha) + R[i + s, j]
                    )

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, self.max_iter
