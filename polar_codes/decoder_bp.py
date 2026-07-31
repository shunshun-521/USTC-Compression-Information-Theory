"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, _permute_llr
from encoder import polar_encode
from channel import hard_decision_llr

_LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _permute_llr(llr_raw, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                span = 1 << (n - stage + 1)
                half = span // 2
                for block in range(0, N, span):
                    for j in range(half):
                        i = block + j
                        ip = i + half
                        L[i, stage - 1] = self._f_minsum(
                            R[i, stage] + L[ip, stage], L[i, stage]
                        )
                        L[ip, stage - 1] = self._f_minsum(
                            R[i, stage], L[i, stage]
                        ) + L[ip, stage]

            for stage in range(1, n + 1):
                span = 1 << (n - stage + 1)
                half = span // 2
                for block in range(0, N, span):
                    for j in range(half):
                        i = block + j
                        ip = i + half
                        R[i, stage] = self._f_minsum(
                            R[ip, stage] + L[ip, stage], R[i, stage - 1]
                        )
                        R[ip, stage] = self._f_minsum(
                            R[i, stage - 1], L[i, stage]
                        ) + R[ip, stage]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_raw)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat.astype(int), num_iters
