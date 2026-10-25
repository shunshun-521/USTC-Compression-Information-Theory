"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = int(N)
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        Li = R[idx, j - 1] + L[idx + s, j]
                        Lj = L[idx + s, j]
                        L[idx, j - 1] = _f_min_sum(Li, Lj, self.alpha)
                        Ri = R[idx, j - 1]
                        Lk = L[idx, j]
                        Rj = R[idx + s, j]
                        L[idx + s, j - 1] = _f_min_sum(Ri, Lk, self.alpha) + Rj

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        Rk = R[idx + s, j]
                        Lk = L[idx + s, j + 1]
                        Ri = R[idx, j]
                        R[idx, j + 1] = _f_min_sum(Rk + Lk, Ri, self.alpha)
                        Rj = R[idx + s, j]
                        Lk2 = L[idx, j + 1]
                        R[idx + s, j + 1] = _f_min_sum(Ri, Lk2, self.alpha) + Rj

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard)
