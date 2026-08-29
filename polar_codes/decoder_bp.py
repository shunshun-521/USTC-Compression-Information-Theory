"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            L_new = L.copy()
            for stage in range(n - 1, -1, -1):
                s = 2 ** stage
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        L_new[u, stage] = _f_min_sum(
                            R[u, stage + 1] + L[l, stage + 1],
                            L[u, stage + 1],
                            alpha,
                        )
                        L_new[l, stage] = _f_min_sum(
                            R[u, stage + 1],
                            L[u, stage + 1],
                            alpha,
                        ) + L[l, stage + 1]
            L = L_new
            L[:, n] = llr_ch

            R_new = R.copy()
            for stage in range(0, n):
                s = 2 ** stage
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        R_new[u, stage + 1] = _f_min_sum(
                            R[l, stage + 1] + L[l, stage + 1],
                            R[u, stage],
                            alpha,
                        )
                        R_new[l, stage + 1] = _f_min_sum(
                            R[u, stage],
                            L[u, stage + 1],
                            alpha,
                        ) + R[l, stage + 1]
            R = R_new
            R[self.frozen_idx, 0] = self.large

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
