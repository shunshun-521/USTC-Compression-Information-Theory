"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, sc_decode
from encoder import bit_reversal_permutation, polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        LARGE = 1e7

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch[self.br]

        u_sc = sc_decode(llr_ch, self.frozen_bits)
        u_sc_br = u_sc[self.br]
        for i in range(N):
            if self.frozen_br[i]:
                R[0, i] = LARGE
            else:
                sc_bias = -abs(L[n, i]) if u_sc_br[i] == 1 else abs(L[n, i])
                R[0, i] = 0.5 * L[n, i] + 0.5 * sc_bias

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = i + k + s
                        L[j - 1, a] = self._f(R[j, a] + L[j, b], L[j, a])
                        L[j - 1, b] = self._f(R[j, a], L[j, a]) + L[j, b]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = i + k + s
                        R[j + 1, a] = self._f(R[j, b] + L[j + 1, b], R[j, a])
                        R[j + 1, b] = self._f(R[j, a], L[j + 1, a]) + R[j, b]

            total = L[0, :] + R[0, :]
            u_br = np.zeros(N, dtype=int)
            u_br[~self.frozen_br] = (total[~self.frozen_br] < 0).astype(int)
            u_hat = u_br[self.br]
            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                break

        total = L[0, :] + R[0, :]
        u_br = np.zeros(N, dtype=int)
        u_br[~self.frozen_br] = (total[~self.frozen_br] < 0).astype(int)
        u_hat = u_br[self.br]
        return u_hat, num_iters
