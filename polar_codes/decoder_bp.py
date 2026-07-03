"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _reorder_channel_llrs


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _ms_f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_received = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _reorder_channel_llrs(llr_received)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    L[i, j - 1] = self._ms_f(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = self._ms_f(R[i, j], L[i, j]) + L[i + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    R[i, j] = self._ms_f(
                        R[i + step, j] + L[i + step, j], R[i, j - 1]
                    )
                    R[i + step, j] = self._ms_f(R[i, j - 1], L[i, j]) + R[i + step, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_received):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_received):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_received < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
