"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（Parizi & Karzand 消息传递形式）"""

    LARGE = 1e10

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha
        half = N // 2

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n] = llr_ch[self.br]
        R[0] = 0.0
        R[0, self.frozen_bits == 1] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n, 0, -1):
                for j in range(half):
                    L[i - 1, 2 * j] = _f_min_sum(
                        R[i, j] + L[i, j + half], L[i, j], alpha
                    )
                    L[i - 1, 2 * j + 1] = _f_min_sum(
                        R[i, j], L[i, j], alpha
                    ) + L[i, j + half]

            for i in range(1, n + 1):
                for j in range(half):
                    R[i, 2 * j] = _f_min_sum(
                        R[i - 1, j + half] + L[i, j + half], R[i - 1, j], alpha
                    )
                    R[i, 2 * j + 1] = _f_min_sum(
                        R[i - 1, j], L[i, j + half], alpha
                    ) + R[i - 1, j + half]

            for k in range(N):
                total = L[0, k] + R[0, k]
                if self.frozen_bits[k]:
                    u_hat[k] = 0
                else:
                    u_hat[k] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for k in range(N):
            total = L[0, k] + R[0, k]
            if self.frozen_bits[k]:
                u_hat[k] = 0
            else:
                u_hat[k] = 0 if total >= 0 else 1

        return u_hat, num_iters
