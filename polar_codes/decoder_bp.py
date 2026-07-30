"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _prepare_llr


def _bp_f(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


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
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        llr = _prepare_llr(llr_ch)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = R[i, j - 1] + L[i + s, j]
                    Lj = L[i, j]
                    Ls = L[i + s, j]
                    Ri = R[i, j - 1]
                    L[i, j - 1] = _bp_f(Li, Lj, alpha)
                    L[i + s, j - 1] = _bp_f(Ri, Lj, alpha) + Ls

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Rs = R[i + s, j]
                    Ls = L[i + s, j + 1]
                    Ri = R[i, j - 1] if j > 0 else R[i, 0]
                    Lj = L[i, j + 1]
                    R[i, j] = _bp_f(Rs + Ls, Ri, alpha)
                    R[i + s, j] = _bp_f(Ri, Lj, alpha) + Rs

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
