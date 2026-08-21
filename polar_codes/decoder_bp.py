"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _hard_decision(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(i, i + step):
                        L[k, j - 1] = _ms_f(
                            R[k, j - 1] + L[k + step, j],
                            L[k, j],
                            self.alpha,
                        )
                        L[k + step, j - 1] = _ms_f(
                            R[k, j - 1],
                            L[k, j],
                            self.alpha,
                        ) + L[k + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(i, i + step):
                        R[k, j] = _ms_f(
                            R[k + step, j - 1] + L[k + step, j],
                            R[k, j - 1],
                            self.alpha,
                        )
                        R[k + step, j - 1] = _ms_f(
                            R[k, j - 1],
                            L[k, j],
                            self.alpha,
                        ) + R[k + step, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.frozen_idx] = 0
            info_idx = np.where(self.frozen_bits == 0)[0]
            u_hat[info_idx] = (total[info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_decision(llr_ch)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.frozen_idx] = 0
        info_idx = np.where(self.frozen_bits == 0)[0]
        u_hat[info_idx] = (total[info_idx] < 0).astype(int)
        return u_hat, num_iters
