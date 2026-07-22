"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器（因子图，参考 ICECCME 2023 公式 3-6）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 2, N), dtype=np.float64)
        R = np.zeros((n + 2, N), dtype=np.float64)

        L[n + 1, :] = llr_ch
        R[1, self.frozen_bits] = self.large
        R[1, ~self.frozen_bits] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                offset = 1 << (n - stage)
                for i in range(0, N, 2 * offset):
                    j = i + offset
                    L[stage, i] = _f_min_sum(
                        L[stage + 1, i],
                        L[stage + 1, j] + R[stage, j],
                        self.alpha,
                    )
                    L[stage, j] = _f_min_sum(
                        R[stage, i],
                        L[stage + 1, i],
                        self.alpha,
                    ) + L[stage + 1, j]

            for stage in range(1, n + 1):
                offset = 1 << (n - stage)
                for i in range(0, N, 2 * offset):
                    j = i + offset
                    R[stage + 1, i] = _f_min_sum(
                        R[stage, j] + L[stage + 1, j],
                        R[stage, i],
                        self.alpha,
                    )
                    R[stage + 1, j] = _f_min_sum(
                        R[stage, i],
                        L[stage + 1, i],
                        self.alpha,
                    ) + R[stage, j]

            posterior = L[1, :] + R[1, :]
            u_hat = np.where(posterior >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        posterior = L[1, :] + R[1, :]
        u_hat = np.where(posterior >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
