"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 L/R 消息传递，列 0=信源，列 n=信道）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = i + k + s
                        L[a, j] = _ms_f(
                            R[a, j + 1] + L[b, j + 1],
                            L[a, j + 1],
                            self.alpha,
                        )
                        L[b, j] = _ms_f(
                            R[a, j + 1],
                            L[a, j + 1],
                            self.alpha,
                        ) + L[b, j + 1]

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = i + k + s
                        R[a, j + 1] = _ms_f(
                            R[b, j] + L[b, j + 1],
                            R[a, j],
                            self.alpha,
                        )
                        R[b, j + 1] = _ms_f(
                            R[a, j],
                            L[a, j + 1],
                            self.alpha,
                        ) + R[b, j]

            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
