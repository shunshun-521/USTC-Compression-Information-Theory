"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(x, y, alpha):
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
        self._large = 1e6

    def _hard_codeword(self, llr_ch):
        x_hat = (llr_ch < 0).astype(int)
        return x_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = _f_minsum(
                            R[idx_u, j] + L[idx_l, j],
                            L[idx_u, j],
                            self.alpha,
                        )
                        L[idx_l, j - 1] = (
                            _f_minsum(R[idx_u, j], L[idx_u, j], self.alpha)
                            + L[idx_l, j]
                        )

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_l, j] = _f_minsum(
                            R[idx_u, j - 1],
                            L[idx_l, j],
                            self.alpha,
                        ) + R[idx_l, j - 1]
                        R[idx_u, j] = _f_minsum(
                            R[idx_l, j] + L[idx_l, j],
                            R[idx_u, j - 1],
                            self.alpha,
                        )

            u_hat = (L[:, 0] + R[:, 0] >= 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_codeword(llr_ch)):
                num_iters = it
                return u_hat, num_iters

            num_iters = it

        u_hat = (L[:, 0] + R[:, 0] >= 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
