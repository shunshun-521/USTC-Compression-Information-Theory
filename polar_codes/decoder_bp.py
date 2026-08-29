"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation


def bp_f(x, y, alpha=0.9375):
    """min-sum f 运算"""
    if isinstance(x, np.ndarray):
        return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))
    sx, sy = np.sign(x), np.sign(y)
    if sx == 0:
        sx = 1
    if sy == 0:
        sy = 1
    return alpha * sx * sy * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        llr = llr_ch[self.br]

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        L[idx, j - 1] = bp_f(
                            R[idx, j - 1] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = bp_f(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[idx, j + 1] = bp_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = bp_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx2, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0
            num_iters = self.max_iter

        return u_hat, num_iters
