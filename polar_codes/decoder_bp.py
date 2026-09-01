"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _recover_u(self, L, R):
        """从码字列后验 LLR 恢复信息序列 u"""
        x_hat = (L[:, self.n] + R[:, self.n] < 0).astype(int)
        u_hat = polar_encode(x_hat)
        u_hat[self.frozen_idx] = 0
        return u_hat, x_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        hard_ch = (llr_ch < 0).astype(int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i, j + 1]
                    Li = L[i, j + 1]
                    Lis = L[i + s, j + 1]
                    L[i, j] = _f_min_sum(Ri + Lis, Li, self.alpha)
                    L[i + s, j] = _f_min_sum(Ri, Li, self.alpha) + Lis

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i, j]
                    Ris = R[i + s, j]
                    Li = L[i, j + 1]
                    Lis = L[i + s, j + 1]
                    R[i, j + 1] = _f_min_sum(Ris + Lis, Ri, self.alpha)
                    R[i + s, j + 1] = _f_min_sum(Ri, Li, self.alpha) + Ris

            u_hat, x_hat = self._recover_u(L, R)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat, _ = self._recover_u(L, R)
        return u_hat, num_iters
