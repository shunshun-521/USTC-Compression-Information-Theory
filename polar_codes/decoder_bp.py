"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def bp_f(x, y, alpha=0.9375):
    """min-sum f 函数"""
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.br_inv = np.argsort(self.br)

        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((N, self.n + 1), dtype=np.float64)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br]

        self.L[:, self.n] = llr_perm
        self.R[:, 0] = 0.0
        for i in range(self.N):
            if self.frozen_bits[i]:
                self.R[i, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            self._update_L()
            self._update_R()
            num_iters = it

            u_hat = self._hard_decision()
            if self._early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision()
        return u_hat, num_iters

    def _update_L(self):
        for j in range(self.n, 0, -1):
            step = 1 << (j - 1)
            for i in range(0, self.N, step * 2):
                for k in range(step):
                    idx = i + k
                    s = step
                    self.L[idx, j - 1] = bp_f(
                        self.R[idx, j] + self.L[idx + s, j],
                        self.L[idx, j],
                        self.alpha,
                    )
                    self.L[idx + s, j - 1] = bp_f(
                        self.R[idx, j],
                        self.L[idx, j],
                        self.alpha,
                    ) + self.L[idx + s, j]

    def _update_R(self):
        for j in range(1, self.n + 1):
            step = 1 << (j - 1)
            for i in range(0, self.N, step * 2):
                for k in range(step):
                    idx = i + k
                    s = step
                    self.R[idx, j] = bp_f(
                        self.R[idx + s, j] + self.L[idx + s, j],
                        self.R[idx, j - 1],
                        self.alpha,
                    )
                    self.R[idx + s, j] = bp_f(
                        self.R[idx, j - 1],
                        self.L[idx, j],
                        self.alpha,
                    ) + self.R[idx + s, j]

    def _hard_decision(self):
        u_hat = np.zeros(self.N, dtype=int)
        total = self.L[:, 0] + self.R[:, 0]
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_bits = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_bits)
