"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _bp_f(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i][0] + R[i][0]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = [[0.0] * (n + 1) for _ in range(N)]
        R = [[0.0] * (n + 1) for _ in range(N)]

        for i in range(N):
            L[i][n] = float(llr_ch[i])
            R[i][0] = 0.0 if not self.frozen_bits[i] else self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        L[idx][j - 1] = _bp_f(
                            R[idx][j] + L[idx2][j],
                            L[idx][j],
                            alpha,
                        )
                        L[idx2][j - 1] = (
                            _bp_f(R[idx][j], L[idx][j], alpha) + L[idx2][j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        R[idx][j + 1] = _bp_f(
                            R[idx2][j] + L[idx2][j + 1],
                            R[idx][j],
                            alpha,
                        )
                        R[idx2][j + 1] = (
                            _bp_f(R[idx][j], L[idx][j + 1], alpha) + R[idx2][j]
                        )

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
