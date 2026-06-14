"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_llr
from encoder import polar_encode

_LARGE = 1e6


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr_ch = _prepare_llr(llr_ch)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_s = i + k + s
                        L[idx_i, j - 1] = _f_min_sum(
                            R[idx_i, j] + L[idx_s, j],
                            L[idx_i, j],
                            self.alpha,
                        )
                        L[idx_s, j - 1] = _f_min_sum(
                            R[idx_i, j],
                            L[idx_i, j],
                            self.alpha,
                        ) + L[idx_s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_s = i + k + s
                        R[idx_i, j + 1] = _f_min_sum(
                            R[idx_s, j] + L[idx_s, j + 1],
                            R[idx_i, j],
                            self.alpha,
                        )
                        R[idx_s, j + 1] = _f_min_sum(
                            R[idx_i, j],
                            L[idx_s, j + 1],
                            self.alpha,
                        ) + R[idx_s, j]

            u_hat = self._hard_decision_from_lr(L, R)
            x_hat = polar_encode(u_hat)
            hard_natural = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_natural):
                break

        u_hat = self._hard_decision_from_lr(L, R)
        return u_hat, num_iters

    def _hard_decision_from_lr(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat
