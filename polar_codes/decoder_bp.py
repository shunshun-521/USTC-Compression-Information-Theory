"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=float)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.LARGE

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j - 1] = _ms_f(R[idx_u, j - 1] + L[idx_l, j], L[idx_u, j], self.alpha)
                        L[idx_l, j - 1] = _ms_f(R[idx_u, j - 1], L[idx_u, j], self.alpha) + L[idx_l, j]

            # 从左到右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j] = _ms_f(
                            R[idx_l, j] + L[idx_l, j], R[idx_u, j - 1], self.alpha
                        )
                        R[idx_l, j] = _ms_f(R[idx_u, j - 1], L[idx_u, j], self.alpha) + R[idx_l, j - 1]

            u_hat = self._hard_decision(L, R)

            if self._early_stop(u_hat, llr_ch):
                return u_hat, it

        u_hat = self._hard_decision(L, R)
        return u_hat, self.max_iter

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard)
