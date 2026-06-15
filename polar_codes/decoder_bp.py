"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _bp_f(x, y, alpha):
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
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr[self.rev]

        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _bp_f(
                        R[i, j] + L[i + s, j], L[i, j], self.alpha
                    )
                    L[i + s, j - 1] = _bp_f(R[i, j], L[i, j], self.alpha) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _bp_f(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = _bp_f(R[i, j], L[i + s, j + 1], self.alpha) + R[
                        i + s, j
                    ]

            u_hat = np.zeros(N, dtype=np.int8)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_indices] = 0
        return u_hat.astype(int), num_iters
