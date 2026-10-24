"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _upper_llr_scalar


class BPDecoder:
    """BP 译码器（Sionna/Arikan 因子图索引）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        ind_range = np.arange(N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
            ind_2 = ind_1 + 2**ind_s
            self.stage_ind_1.append(ind_1.astype(int))
            self.stage_ind_2.append(ind_2.astype(int))

    def _f(self, x, y):
        if self.alpha == 1.0:
            return _upper_llr_scalar(float(x), float(y))
        return self.alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n):
                i1 = self.stage_ind_1[s]
                i2 = self.stage_ind_2[s]
                for k in range(len(i1)):
                    a, b = i1[k], i2[k]
                    R[a, s + 1] = self._f(R[b, s] + L[b, s + 1], R[a, s])
                    R[b, s + 1] = self._f(R[a, s], L[a, s + 1]) + R[b, s]

            for s in range(n - 1, -1, -1):
                i1 = self.stage_ind_1[s]
                i2 = self.stage_ind_2[s]
                for k in range(len(i1)):
                    a, b = i1[k], i2[k]
                    L[a, s] = self._f(L[a, s + 1], L[b, s + 1] + R[b, s + 1])
                    L[b, s] = self._f(R[a, s + 1], L[a, s + 1]) + L[b, s + 1]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        return self._hard_decision(L, R), num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat
