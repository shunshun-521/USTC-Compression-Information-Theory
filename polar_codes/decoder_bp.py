"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import _permute_llr_for_decoder


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr = _permute_llr_for_decoder(llr_nat)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        for it in range(self.max_iter):
            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        u = block + i
                        v = u + stride
                        L[u, stage] = _f_minsum(
                            R[u, stage + 1] + L[v, stage + 1], L[u, stage + 1], self.alpha
                        )
                        L[v, stage] = (
                            _f_minsum(R[u, stage + 1], L[u, stage + 1], self.alpha)
                            + L[v, stage + 1]
                        )

            for stage in range(0, n):
                stride = 1 << stage
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        u = block + i
                        v = u + stride
                        R[u, stage + 1] = _f_minsum(
                            R[v, stage] + L[v, stage + 1], R[u, stage], self.alpha
                        )
                        R[v, stage + 1] = (
                            _f_minsum(R[u, stage], L[u, stage + 1], self.alpha) + R[v, stage]
                        )

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_nat)):
                return u_hat, it + 1

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, self.max_iter
