"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import permute_llr_for_decode
from encoder import polar_encode


def bp_f(x, y, alpha=0.9375):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），列 0 为信源，列 n 为信道。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = permute_llr_for_decode(llr_orig)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L_old = L.copy()
            R_old = R.copy()

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = bp_f(
                            R_old[idx, j] + L_old[idx + s, j],
                            L_old[idx, j],
                            alpha,
                        )
                        L[idx + s, j - 1] = bp_f(
                            R_old[idx, j], L_old[idx, j], alpha
                        ) + L_old[idx + s, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = bp_f(
                            R_old[idx + s, j] + L_old[idx + s, j],
                            R_old[idx, j - 1],
                            alpha,
                        )
                        R[idx + s, j] = bp_f(
                            R_old[idx, j - 1], L_old[idx + s, j], alpha
                        ) + R_old[idx + s, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_orig)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
