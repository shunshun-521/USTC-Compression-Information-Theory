"""
极化码 BP（置信传播）译码器
基于因子图（n+1 列）的 min-sum BP，含早停；未收敛时回退 SC 以保证鲁棒性
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _bp_iterate(self, llr_ch):
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    ri = i + s
                    L[i, j - 1] = _minsum_f(
                        R[i, j] + L[ri, j], L[i, j], self.alpha
                    )
                    L[ri, j - 1] = (
                        _minsum_f(R[i, j], L[i, j], self.alpha) + L[ri, j]
                    )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    ri = i + s
                    R[i, j] = _minsum_f(
                        R[ri, j] + L[ri, j], R[i, j - 1], self.alpha
                    )
                    R[ri, j] = (
                        _minsum_f(R[i, j - 1], L[i, j], self.alpha) + R[ri, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters

    def decode(self, llr_ch):
        """主译码：BP 迭代；若码字校验失败则回退 SC。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat, num_iters = self._bp_iterate(llr_ch)

        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        if np.array_equal(x_hat, hard_ch):
            return u_hat, num_iters

        from decoder_sc import sc_decode

        return sc_decode(llr_ch, self.frozen_bits), num_iters
