"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation

LARGE = 1e6


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器（Kaira/Arikan 因子图消息传递结构）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _update_left(self, R, L):
        m, N, alpha = self.n, self.N, self.alpha
        for i in range(m - 1, -1, -1):
            add_k = N // (2 ** (m - i))
            for base in range(0, N, 2 * add_k):
                for off in range(add_k):
                    left = base + off
                    right = left + add_k
                    L[i, left] = _minsum_f(
                        L[i + 1, left], L[i + 1, right] + R[i, right], alpha
                    )
                    L[i, right] = (
                        _minsum_f(R[i, left], L[i + 1, left], alpha) + L[i + 1, right]
                    )
        return L

    def _update_right(self, R, L):
        m, N, alpha = self.n, self.N, self.alpha
        for i in range(m):
            add_k = N // (2 ** (m - i))
            for base in range(0, N, 2 * add_k):
                for off in range(add_k):
                    left = base + off
                    right = left + add_k
                    R[i + 1, left] = _minsum_f(
                        R[i, left], L[i + 1, right] + R[i, right], alpha
                    )
                    R[i + 1, right] = (
                        _minsum_f(R[i, left], L[i + 1, left], alpha) + R[i, right]
                    )
        return R

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m, N = self.n, self.N

        R = np.zeros((m + 1, N), dtype=np.float64)
        L = np.zeros((m + 1, N), dtype=np.float64)
        L[m, :] = llr_ch
        R[0, self.frozen_bits] = LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L = self._update_left(R, L)
            R = self._update_right(R, L)

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
