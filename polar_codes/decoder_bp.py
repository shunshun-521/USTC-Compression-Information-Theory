"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _g_minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        rev = bit_reversal_permutation(N)
        llr = llr_ch[rev].copy()

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr
        R[0] = 0.0
        for idx in self.frozen_indices:
            R[0, idx] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 从右到左更新 L
            for s in range(n - 1, -1, -1):
                block = 2 ** s
                for j in range(0, N, 2 * block):
                    for k in range(block):
                        L[s, j + k] = self._g_minsum(
                            L[s + 1, j + k],
                            L[s + 1, j + k + block] + R[s, j + k + block],
                        )
                        L[s, j + k + block] = self._g_minsum(
                            L[s, j + k], L[s + 1, j + k]
                        ) + L[s + 1, j + k + block]

            # 从左到右更新 R
            for s in range(0, n):
                block = 2 ** s
                for j in range(0, N, 2 * block):
                    for k in range(block):
                        R[s + 1, j + k] = self._g_minsum(
                            R[s, j + k + block] + L[s + 1, j + k + block],
                            R[s, j + k],
                        )
                        R[s + 1, j + k + block] = self._g_minsum(
                            R[s, j + k], L[s + 1, j + k]
                        ) + R[s, j + k + block]

            # 早停检查
            total_llr = L[0] + R[0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_indices:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it + 1
                break

        total_llr = L[0] + R[0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if i in self.frozen_indices:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1

        return u_hat, num_iters
