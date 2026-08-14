"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _minsum_f(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算"""
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
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        n = self.n
        N = self.N
        brp = bit_reversal_permutation(N)
        llr_ch = llr_ch[brp]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch.copy()
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_l = i + k
                        idx_r = i + k + s
                        L[idx_l, j] = _minsum_f(
                            R[idx_l, j] + L[idx_r, j + 1],
                            L[idx_l, j + 1], self.alpha)
                        L[idx_r, j] = (
                            _minsum_f(R[idx_l, j], L[idx_l, j + 1], self.alpha)
                            + L[idx_r, j + 1])

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_l = i + k
                        idx_r = i + k + s
                        R[idx_l, j + 1] = _minsum_f(
                            R[idx_r, j] + L[idx_r, j + 1],
                            R[idx_l, j], self.alpha)
                        R[idx_r, j + 1] = (
                            _minsum_f(R[idx_l, j], L[idx_l, j + 1], self.alpha)
                            + R[idx_r, j])

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
