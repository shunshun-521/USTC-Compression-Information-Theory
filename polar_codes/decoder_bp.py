"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import channel_llr_to_decoder


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((N, self.n + 1), dtype=np.float64)

    def _update_L(self):
        for j in range(self.n - 1, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, self.N, 2 * s):
                for k in range(s):
                    idx = i + k
                    self.L[idx, j - 1] = _f_min_sum(
                        self.R[idx, j] + self.L[idx + s, j + 1],
                        self.L[idx, j + 1],
                        self.alpha,
                    )
                    self.L[idx + s, j - 1] = (
                        _f_min_sum(self.R[idx, j], self.L[idx, j + 1], self.alpha)
                        + self.L[idx + s, j + 1]
                    )

    def _update_R(self):
        for j in range(0, self.n):
            s = 1 << j
            for i in range(0, self.N, 2 * s):
                for k in range(s):
                    idx = i + k
                    self.R[idx, j + 1] = _f_min_sum(
                        self.R[idx + s, j] + self.L[idx + s, j + 1],
                        self.R[idx, j],
                        self.alpha,
                    )
                    self.R[idx + s, j + 1] = (
                        _f_min_sum(self.R[idx, j], self.L[idx, j + 1], self.alpha)
                        + self.R[idx + s, j]
                    )

    def _hard_decision(self):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            total = self.L[i, 0] + self.R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat

    def _early_stop(self, llr_ch):
        u_hat = self._hard_decision()
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = channel_llr_to_decoder(llr_ch)

        self.L.fill(0.0)
        self.R.fill(0.0)
        self.L[:, self.n] = llr_dec
        self.R[:, 0] = 0.0
        for i in range(self.N):
            if self.frozen_bits[i]:
                self.R[i, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            self._update_L()
            self._update_R()
            num_iters = it
            if self._early_stop(llr_ch):
                break

        return self._hard_decision(), num_iters
