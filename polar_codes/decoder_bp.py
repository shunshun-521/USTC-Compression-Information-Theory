"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        br = bit_reversal_permutation(N)
        L[:, n] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for phi in range(n - 1, -1, -1):
                step = 1 << phi
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        L[a, phi] = _f_min_sum(R[a, phi + 1] + L[b, phi + 1], L[a, phi + 1], alpha)
                        L[b, phi] = _f_min_sum(R[a, phi + 1], L[a, phi + 1], alpha) + L[b, phi + 1]

            for phi in range(n):
                step = 1 << phi
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        R[a, phi + 1] = _f_min_sum(R[b, phi] + L[b, phi + 1], R[a, phi], alpha)
                        R[b, phi + 1] = _f_min_sum(R[a, phi], L[b, phi + 1], alpha) + R[b, phi]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
