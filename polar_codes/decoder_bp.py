"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_int
from channel import hard_decision_llr
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（分层因子图，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = frozen_bits
        else:
            self.frozen = frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        br_idx = np.array([bit_reversal_int(i, n) for i in range(N)])
        L[n, :] = llr_ch[br_idx]
        R[0, :] = 0.0
        R[0, self.frozen[br_idx]] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                s = 1 << (layer - 1)
                count = 1 << (n - layer)
                for block in range(count):
                    base = block * (2 * s)
                    for i in range(s):
                        idx = base + i
                        L[layer - 1, idx] = self._ms(
                            R[layer, idx] + L[layer, idx + s],
                            L[layer, idx],
                        )
                        L[layer - 1, idx + s] = self._ms(
                            R[layer, idx],
                            L[layer, idx],
                        ) + L[layer, idx + s]

            for layer in range(0, n):
                s = 1 << layer
                count = 1 << (n - layer - 1)
                for block in range(count):
                    base = block * (2 * s)
                    for i in range(s):
                        idx = base + i
                        R[layer + 1, idx] = self._ms(
                            R[layer + 1, idx + s] + L[layer + 1, idx + s],
                            R[layer, idx],
                        )
                        R[layer + 1, idx + s] = (
                            self._ms(R[layer, idx], L[layer + 1, idx])
                            + R[layer + 1, idx + s]
                        )

            total_br = L[0, :] + R[0, :]
            u_hat_br = (total_br < 0).astype(int)
            u_hat_br[self.frozen[br_idx]] = 0

            u_hat = np.zeros(N, dtype=int)
            u_hat[br_idx] = u_hat_br

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total_br = L[0, :] + R[0, :]
        u_hat_br = (total_br < 0).astype(int)
        u_hat_br[self.frozen[br_idx]] = 0
        u_hat = np.zeros(N, dtype=int)
        u_hat[br_idx] = u_hat_br
        return u_hat, num_iters
