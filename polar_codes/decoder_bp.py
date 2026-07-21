"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N = self.N
        n = self.n
        alpha = self.alpha
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx + s, j], L[idx, j], alpha
                        )
                        L[idx + s, j - 1] = _minsum_f(
                            R[idx, j], L[idx, j], alpha
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _minsum_f(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j], alpha
                        )
                        R[idx + s, j + 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], alpha
                        ) + R[idx + s, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                num_iters = it
                break
            num_iters = it

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
