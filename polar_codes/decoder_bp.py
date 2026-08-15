"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


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
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.bit_rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                step = 2 ** (layer - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        L[a, layer - 1] = _f_minsum(
                            R[a, layer] + L[b, layer], L[a, layer], self.alpha
                        )
                        L[b, layer - 1] = _f_minsum(
                            R[a, layer], L[a, layer], self.alpha
                        ) + L[b, layer]

            for layer in range(0, n):
                step = 2 ** layer
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        R[a, layer + 1] = _f_minsum(
                            R[b, layer] + L[b, layer + 1], R[a, layer], self.alpha
                        )
                        R[b, layer + 1] = _f_minsum(
                            R[a, layer], L[a, layer + 1], self.alpha
                        ) + R[b, layer]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)[self.bit_rev]
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits.astype(bool)] = 0

        return u_hat.astype(int), num_iters
