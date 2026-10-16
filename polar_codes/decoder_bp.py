"""
极化码 BP（置信传播）译码器
基于因子图 min-sum 近似；早停时与 SC 一致校验重编码结果
"""
import math
import numpy as np
from decoder_sc import f_operation, sc_decode
from encoder import polar_encode, bit_reversal_permutation

LARGE = 1e6


class BPDecoder:
    """BP 译码器（阶段域消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._rev = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self._rev].copy()
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr
        R[0, self.frozen_idx] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        L[s, idx] = self._f_ms(
                            L[s + 1, idx] + R[s + 1, idx],
                            L[s + 1, idx + step],
                        )
                        L[s, idx + step] = self._f_ms(
                            R[s + 1, idx], L[s + 1, idx]
                        ) + L[s + 1, idx + step]

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        R[s + 1, idx] = self._f_ms(
                            R[s + 1, idx + step] + L[s + 1, idx + step],
                            R[s, idx],
                        )
                        R[s + 1, idx + step] = self._f_ms(
                            R[s, idx], L[s + 1, idx]
                        ) + R[s + 1, idx + step]

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            u_nat = np.zeros(N, dtype=int)
            u_nat[self._rev] = u_hat
            x_hat = polar_encode(u_nat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_nat, num_iters

        u_sc = sc_decode(llr_ch, self.frozen_bits)
        return u_sc, num_iters
