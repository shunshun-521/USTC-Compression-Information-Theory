"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, j - 1] = _f_min_sum(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _f_min_sum(R[i0, j], L[i0, j], self.alpha) + L[i1, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, j] = _f_min_sum(
                            R[i1, j] + L[i1, j], R[i0, j - 1], self.alpha
                        )
                        R[i1, j] = _f_min_sum(R[i0, j - 1], L[i0, j], self.alpha) + R[i1, j]

            u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
