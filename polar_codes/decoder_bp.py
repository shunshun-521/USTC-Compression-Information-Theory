"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _frozen_to_set, _prepare_llr


class BPDecoder:
    """BP 译码器（Hashemi 因子图 + min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e7

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _g(self, a, b, c):
        return self._f(a, b) + c

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        R[:, 0] = 0.0

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            L[:, n] = llr.copy()
            for idx in self.frozen_set:
                L[idx, n] = llr[idx] + self._large

            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for block in range(0, N, 2 * step):
                    for w in range(step):
                        i = block + w
                        j = i + step
                        L[i, lam] = self._f(R[i, lam] + L[i, lam + 1], L[j, lam + 1])
                        L[j, lam] = self._g(R[i, lam], L[i, lam + 1], L[j, lam + 1] + R[j, lam])

            for lam in range(0, n):
                step = 1 << lam
                for block in range(0, N, 2 * step):
                    for w in range(step):
                        i = block + w
                        j = i + step
                        R[i, lam + 1] = self._g(R[j, lam + 1], L[j, lam + 1], R[i, lam])
                        R[j, lam + 1] = self._f(R[i, lam], L[i, lam + 1]) + R[j, lam]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat, num_iters
