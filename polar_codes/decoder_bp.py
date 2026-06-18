"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


def _minsum_f(a, b, alpha=0.9375):
    """min-sum f 运算，带归一化因子 alpha。"""
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数。"""
        n, N = self.n, self.N
        brp = bit_reversal_permutation(N)
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_raw[brp]
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[j - 1, i] = _minsum_f(
                            R[j, i] + L[j, i + s], L[j, i], self.alpha
                        )
                        L[j - 1, i + s] = _minsum_f(
                            R[j, i], L[j, i]
                        ) + L[j, i + s]

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[j + 1, i] = _minsum_f(
                            R[j + 1, i + s] + L[j + 1, i + s],
                            R[j, i],
                            self.alpha,
                        )
                        R[j + 1, i + s] = _minsum_f(
                            R[j, i], L[j + 1, i]
                        ) + R[j + 1, i + s]

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
