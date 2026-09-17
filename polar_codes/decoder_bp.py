"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算，带归一化因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._br = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_perm = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat[self._br], hard_perm)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L_new = np.zeros((N, n + 1), dtype=np.float64)
        R_new = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L_new[:] = L
            R_new[:] = R

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idxs = idx + s
                        L_new[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[idxs, j], L[idx, j], alpha
                        )
                        L_new[idxs, j - 1] = (
                            _f_min_sum(R[idx, j - 1], L[idx, j], alpha)
                            + L[idxs, j]
                        )

            L[:] = L_new

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idxs = idx + s
                        R_new[idx, j + 1] = _f_min_sum(
                            R[idxs, j] + L[idxs, j + 1], R[idx, j], alpha
                        )
                        R_new[idxs, j + 1] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], alpha)
                            + R[idxs, j]
                        )

            R[:] = R_new

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
