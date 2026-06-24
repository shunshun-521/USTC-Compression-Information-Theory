"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(round(np.log2(N)))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for i in range(0, N, 2 * s):
                    L[i, stage - 1] = _f_minsum(
                        R[i, stage] + L[i + s, stage], L[i, stage], self.alpha
                    )
                    L[i + s, stage - 1] = _f_minsum(
                        R[i, stage], L[i, stage], self.alpha
                    ) + L[i + s, stage]

            for stage in range(1, n + 1):
                s = 1 << (stage - 1)
                for i in range(0, N, 2 * s):
                    R[i, stage] = _f_minsum(
                        R[i + s, stage] + L[i + s, stage], R[i, stage - 1], self.alpha
                    )
                    R[i + s, stage] = _f_minsum(
                        R[i, stage - 1], L[i, stage], self.alpha
                    ) + R[i + s, stage]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
