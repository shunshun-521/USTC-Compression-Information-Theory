"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _minsum_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        hard = (llr_ch < 0).astype(int)
        return hard[self.rev]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        L[idx0, j - 1] = _minsum_f(
                            R[idx0, j] + L[idx1, j],
                            L[idx0, j],
                            self.alpha,
                        )
                        L[idx1, j - 1] = _minsum_f(
                            R[idx0, j],
                            L[idx0, j],
                            self.alpha,
                        ) + L[idx1, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx0 = i + k
                        idx1 = i + k + step
                        R[idx0, j] = _minsum_f(
                            R[idx1, j] + L[idx1, j],
                            R[idx0, j - 1],
                            self.alpha,
                        )
                        R[idx1, j] = _minsum_f(
                            R[idx0, j - 1],
                            L[idx0, j],
                            self.alpha,
                        ) + R[idx1, j]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            hard = self._hard_bits_from_llr(llr_ch)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
