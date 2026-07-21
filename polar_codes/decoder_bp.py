"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _bp_f_ms(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat（自然序）, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        hard_ch = (llr_ch < 0).astype(int)
        llr = llr_ch[self.br]
        frozen_perm = self.frozen_bits[self.br]

        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[frozen_perm, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        L[top, j - 1] = _bp_f_ms(
                            R[top, j] + L[bot, j], L[top, j], alpha
                        )
                        L[bot, j - 1] = _bp_f_ms(R[top, j], L[top, j], alpha) + L[bot, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        R[top, j] = _bp_f_ms(
                            R[bot, j] + L[bot, j], R[top, j - 1], alpha
                        )
                        R[bot, j] = _bp_f_ms(R[top, j - 1], L[top, j], alpha) + R[bot, j]

            num_iters = it
            u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
            u_hat[frozen_perm] = 0

            if np.array_equal(polar_encode(u_hat), hard_ch):
                break

        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[frozen_perm] = 0
        return u_hat, num_iters
