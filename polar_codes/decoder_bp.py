"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _prepare_frozen, _align_channel_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _prepare_frozen(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return (
            self.alpha
            * np.sign(x)
            * np.sign(y)
            * np.minimum(np.abs(x), np.abs(y))
        )

    def decode(self, llr_ch):
        llr_ch = _align_channel_llr(llr_ch, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    La = R[i, j - 1] + L[i + s, j]
                    Lb = L[i, j]
                    L[i, j - 1] = self._f_min_sum(La, Lb)
                    L[i + s, j - 1] = self._f_min_sum(R[i, j - 1], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ra = R[i + s, j] + L[i + s, j + 1]
                    Rb = R[i, j]
                    R[i, j + 1] = self._f_min_sum(Ra, Rb)
                    R[i + s, j + 1] = self._f_min_sum(Rb, L[i + s, j + 1]) + R[i + s, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
