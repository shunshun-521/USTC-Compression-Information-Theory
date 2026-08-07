"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation, _frozen_indices_from_mask


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(_frozen_indices_from_mask(frozen_bits))
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def _f_minsum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.brp]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_br

        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                stride = 2 ** (j - 1)
                for i in range(0, N, 2 * stride):
                    for s in range(stride):
                        idx = i + s
                        La = R[idx, j - 1] + L[idx + stride, j]
                        Lb = L[idx, j]
                        L[idx, j - 1] = self._f_minsum(La, Lb)
                        La2 = R[idx, j - 1]
                        Lb2 = L[idx, j]
                        Lc = L[idx + stride, j]
                        L[idx + stride, j - 1] = self._f_minsum(La2, Lb2) + Lc

            # 从左到右更新 R
            for j in range(1, n + 1):
                stride = 2 ** (j - 1)
                for i in range(0, N, 2 * stride):
                    for s in range(stride):
                        idx = i + s
                        Ra = R[idx + stride, j - 1] + L[idx + stride, j]
                        Rb = R[idx, j - 1]
                        R[idx, j] = self._f_minsum(Ra, Rb)
                        Ra2 = R[idx, j - 1]
                        Rb2 = L[idx, j]
                        Rc = R[idx + stride, j - 1]
                        R[idx + stride, j] = self._f_minsum(Ra2, Rb2) + Rc

            # 早停：硬判决 + 重编码校验
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
