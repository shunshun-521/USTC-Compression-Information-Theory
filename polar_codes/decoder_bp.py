"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def bp_f(x, y, alpha=0.9375):
    """BP 中使用的 min-sum f 运算（带修正因子）"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        llr = llr_ch[self.rev].copy()

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[0, :] = llr
        R[n, :] = 0.0

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                stride = 1 << stage
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = i + j + stride
                        L[stage + 1, a] = bp_f(
                            L[stage, a] + R[stage, a],
                            L[stage, b],
                            alpha
                        )
                        L[stage + 1, b] = bp_f(
                            R[stage, a],
                            L[stage, a],
                            alpha
                        ) + L[stage, b]

            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = i + j
                        b = i + j + stride
                        R[stage, b] = bp_f(
                            R[stage + 1, a] + L[stage + 1, b],
                            R[stage + 1, b],
                            alpha
                        )
                        R[stage, a] = bp_f(
                            R[stage + 1, b],
                            L[stage + 1, a],
                            alpha
                        ) + R[stage + 1, a]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[n, i] + R[n, i]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[n, i] + R[n, i]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
