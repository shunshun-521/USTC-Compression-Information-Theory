"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图 min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        L_msg[:, n] = llr_ch
        R_msg[:, 0] = 0.0
        R_msg[self.frozen_bits == 1, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_v = i + k + step
                        L_msg[idx_u, j - 1] = _f_min_sum(
                            R_msg[idx_u, j] + L_msg[idx_v, j],
                            L_msg[idx_u, j],
                            self.alpha,
                        )
                        L_msg[idx_v, j - 1] = (
                            _f_min_sum(R_msg[idx_u, j], L_msg[idx_u, j], self.alpha)
                            + L_msg[idx_v, j]
                        )

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_v = i + k + step
                        R_msg[idx_u, j] = _f_min_sum(
                            R_msg[idx_v, j] + L_msg[idx_v, j],
                            R_msg[idx_u, j - 1],
                            self.alpha,
                        )
                        R_msg[idx_v, j] = (
                            _f_min_sum(R_msg[idx_u, j - 1], L_msg[idx_u, j], self.alpha)
                            + R_msg[idx_v, j]
                        )

            for i in range(N):
                total = L_msg[i, 0] + R_msg[i, 0]
                if self.frozen_bits[i] == 1:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L_msg[i, 0] + R_msg[i, 0]
            if self.frozen_bits[i] == 1:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
