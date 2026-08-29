"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus_minsum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int8)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, alpha = self.N, self.n, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        iu, iv = i + k, i + k + step
                        L[iu, j - 1] = _boxplus_minsum(
                            R[iu, j] + L[iv, j], L[iu, j], alpha
                        )
                        L[iv, j - 1] = (
                            _boxplus_minsum(R[iu, j], L[iu, j], alpha) + L[iv, j]
                        )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        iu, iv = i + k, i + k + step
                        R[iu, j + 1] = _boxplus_minsum(
                            R[iv, j] + L[iv, j + 1], R[iu, j], alpha
                        )
                        R[iv, j + 1] = (
                            _boxplus_minsum(R[iu, j], L[iu, j + 1], alpha) + R[iv, j]
                        )

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat.astype(int), num_iters
