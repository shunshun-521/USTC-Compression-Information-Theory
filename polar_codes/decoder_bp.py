"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from channel import hard_decision_llr
from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e7

    def _decide(self, L0, R0):
        u_hat = np.zeros(self.N, dtype=np.int8)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L0[i] + R0[i]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha

        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        L[n][:] = llr_ch
        R[0][self.frozen_idx] = self.LARGE

        num_iters = 0
        x_hard = hard_decision_llr(llr_ch)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for lam in range(n, 0, -1):
                step = 1 << (lam - 1)
                for beta in range(0, N, 1 << lam):
                    for omega in range(step):
                        i = beta + omega
                        j = i + step
                        L[lam - 1][i] = _boxplus_min_sum(L[lam][i], L[lam][j], alpha)
                        L[lam - 1][j] = _boxplus_min_sum(R[lam][i], L[lam][i], alpha) + L[lam][j]

            for lam in range(1, n + 1):
                step = 1 << (lam - 1)
                for beta in range(0, N, 1 << lam):
                    for omega in range(step):
                        i = beta + omega
                        j = i + step
                        R[lam][i] = _boxplus_min_sum(R[lam - 1][j], L[lam][j], alpha)
                        R[lam][j] = _boxplus_min_sum(R[lam - 1][i], L[lam][i], alpha) + R[lam - 1][j]

            u_hat = self._decide(L[0], R[0])
            if np.array_equal(polar_encode(u_hat), x_hard):
                break

        return self._decide(L[0], R[0]), num_iters
