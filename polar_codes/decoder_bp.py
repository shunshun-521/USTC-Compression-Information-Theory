"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum 近似的 f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器，因子图 n+1 列（0=信源端，n=信道端）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_indices] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        N, n, alpha = self.N, self.n, self.alpha

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L_new[i, j] = _f_min_sum(
                        R[i, j + 1] + L[i + s, j + 1], L[i, j + 1], alpha
                    )
                    L_new[i + s, j] = (
                        _f_min_sum(R[i, j + 1], L[i, j + 1], alpha)
                        + L[i + s, j + 1]
                    )
            L = L_new

            R_new = R.copy()
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R_new[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R_new[i + s, j + 1] = (
                        _f_min_sum(R[i, j], L[i, j + 1], alpha) + R[i + s, j]
                    )
            R = R_new

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
