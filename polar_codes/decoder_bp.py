"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import prepare_channel_llr, hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.damping = 0.75

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_old = L.copy()
            R_old = R.copy()

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx2, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx2, j - 1] = _minsum_f(
                            R[idx, j],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx2, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        R[idx, j] = _minsum_f(
                            R[idx2, j] + L[idx2, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx2, j] = _minsum_f(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + R[idx2, j]

            L = self.damping * L + (1 - self.damping) * L_old
            R = self.damping * R + (1 - self.damping) * R_old

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            num_iters = self.max_iter

        return u_hat, num_iters
