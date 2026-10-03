"""
极化码 BP（置信传播）译码器
因子图 min-sum，含早停
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（信道 LLR 与 SC 相同索引；内部比特倒序对齐因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 1 << j):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        L[idx, j - 1] = _f_minsum(
                            R[idx, j - 1] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _f_minsum(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 1 << (j + 1)):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[idx, j + 1] = _f_minsum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _f_minsum(R[idx, j], L[idx, j + 1], self.alpha) + R[
                            idx2, j
                        ]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        post = L[:, 0] + R[:, 0]
        u_br = np.zeros(self.N, dtype=np.int8)
        u_br[post < 0] = 1
        u_br[self.frozen_bits[self.br]] = 0
        u = np.zeros(self.N, dtype=np.int8)
        u[self.br] = u_br
        u[self.frozen_bits] = 0
        return u
