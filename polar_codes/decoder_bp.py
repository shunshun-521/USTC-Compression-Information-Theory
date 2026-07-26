"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _prepare_llr


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

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = _prepare_llr(llr_ch)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j] = self._f_min_sum(
                            R[i + k, j] + L[i + k + s, j + 1],
                            L[i + k, j + 1],
                        )
                        L[i + k + s, j] = (
                            self._f_min_sum(R[i + k, j], L[i + k, j + 1])
                            + L[i + k + s, j + 1]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j + 1] = self._f_min_sum(
                            R[i + k + s, j] + L[i + k + s, j + 1],
                            R[i + k, j],
                        )
                        R[i + k + s, j + 1] = (
                            self._f_min_sum(R[i + k, j], L[i + k, j + 1])
                            + R[i + k + s, j]
                        )

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
