"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode_core


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.layers = self.n + 1
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _init_messages(self, llr_ch):
        L = np.zeros((self.N, self.layers), dtype=np.float64)
        R = np.zeros((self.N, self.layers), dtype=np.float64)
        L[:, self.n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE
        return L, R

    def _update_l(self, L, R):
        for j in range(self.n, 0, -1):
            s = 1 << (j - 1)
            for block in range(0, self.N, 2 * s):
                for i in range(block, block + s):
                    L_top = L[i, j]
                    L_bot = L[i + s, j]
                    L[i, j - 1] = _ms_f(L_top, L_bot + R[i + s, j - 1], self.alpha)
                    L[i + s, j - 1] = _ms_f(R[i, j - 1], L_top, self.alpha) + L_bot

    def _update_r(self, L, R):
        for j in range(0, self.n):
            s = 1 << j
            for block in range(0, self.N, 2 * s):
                for i in range(block, block + s):
                    R_bot = R[i + s, j]
                    L_bot = L[i + s, j + 1]
                    R_top = R[i, j]
                    L_top = L[i, j + 1]
                    R[i, j + 1] = _ms_f(R_bot + L_bot, R_top, self.alpha)
                    R[i + s, j + 1] = _ms_f(R_top, L_top, self.alpha) + R_bot

    def _hard_decode(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode_core(u_hat)
        hard_x = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_x)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L, R = self._init_messages(llr_ch)
        u_hat = np.zeros(self.N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            self._update_l(L, R)
            self._update_r(L, R)
            u_hat = self._hard_decode(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        return u_hat, num_iters
