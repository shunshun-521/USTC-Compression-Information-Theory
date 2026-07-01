"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen_mask = fb
            self.frozen_idx = np.where(fb)[0]
            self.info_idx = np.where(~fb)[0]
        else:
            fi = fb.astype(int)
            self.frozen_mask = fi != 0
            self.frozen_idx = np.where(fi)[0]
            self.info_idx = np.where(fi == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

        # 预计算各阶段的节点索引，避免译码时重复构造
        self._left_idx = []
        self._right_idx = []
        for s in range(self.n):
            block = 1 << s
            left_parts = []
            right_parts = []
            for j in range(0, N, 2 * block):
                left_parts.append(np.arange(j, j + block))
                right_parts.append(np.arange(j + block, j + 2 * block))
            self._left_idx.append(np.concatenate(left_parts))
            self._right_idx.append(np.concatenate(right_parts))

    @staticmethod
    def _g(alpha, x, y):
        return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, self.info_idx] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_cur = L[n, :].copy()
            for s in range(n - 1, -1, -1):
                a = self._left_idx[s]
                b = self._right_idx[s]
                La = L[s + 1, a]
                Lb = L[s + 1, b]
                Ra = R[s, a]
                Rb = R[s, b]
                L[s, a] = self._g(alpha, La, Lb + Rb)
                L[s, b] = self._g(alpha, La, Ra) + Lb
                L_cur = L[s, :]

            R_cur = R[0, :].copy()
            for s in range(n):
                a = self._left_idx[s]
                b = self._right_idx[s]
                La = L[s + 1, a]
                Lb = L[s + 1, b]
                Ra = R[s, a]
                Rb = R[s, b]
                R[s + 1, a] = self._g(alpha, Ra, Lb + Rb)
                R[s + 1, b] = self._g(alpha, La, Ra) + Rb
                R_cur = R[s + 1, :]

            total = L[0, :] + R[0, :]
            u_hat[~self.frozen_mask] = (total[~self.frozen_mask] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        total = L[0, :] + R[0, :]
        u_hat[~self.frozen_mask] = (total[~self.frozen_mask] < 0).astype(int)
        u_hat[self.frozen_mask] = 0

        return u_hat, num_iters
