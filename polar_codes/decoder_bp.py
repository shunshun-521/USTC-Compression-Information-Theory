"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考: Yazdani & Ardakani, "Low Complexity Belief Propagation Polar Code Decoder"
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def _ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.m

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for _ in range(self.max_iter):
            num_iters += 1

            for s in range(m - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        left = i + k
                        right = i + k + step
                        L[right, s] = L[right, s + 1] + self._ms(L[left, s + 1], R[left, s])
                        tmp = L[right, s + 1] + R[right, s]
                        L[left, s] = self._ms(L[left, s + 1], tmp)

            for s in range(0, m):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        left = i + k
                        right = i + k + step
                        tmp = L[right, s + 1] + R[right, s]
                        R[left, s + 1] = self._ms(R[left, s], tmp)
                        R[right, s + 1] = R[right, s] + self._ms(L[left, s + 1], R[left, s])

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int8)
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if total[idx] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int8)
        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if total[idx] >= 0 else 1

        return u_hat, num_iters
