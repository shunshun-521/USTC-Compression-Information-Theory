"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, _prepare_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool).ravel()
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        llr = _prepare_llr(llr_ch)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        hard_ch = (llr_ch < 0).astype(np.int8)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        L[a, s] = self._minsum(R[a, s + 1] + L[b, s + 1], L[a, s + 1])
                        L[b, s] = self._minsum(R[a, s + 1], L[a, s + 1]) + L[b, s + 1]

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        R[a, s + 1] = self._minsum(R[b, s] + L[b, s + 1], R[a, s])
                        R[b, s + 1] = self._minsum(R[a, s], L[a, s + 1]) + R[b, s]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
