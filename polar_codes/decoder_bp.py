"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
  """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum_f(self, a, b):
        """Min-sum f operation with scaling factor alpha."""
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        LARGE = 1e6

        # L[i,j]: left message at node (i,j), j=0..n
        # R[i,j]: right message at node (i,j), j=0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            # Right-to-left: update L messages at columns n-1 down to 0
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        idx_s = block + i + s
                        L[idx, j] = self._minsum_f(
                            R[idx, j + 1] + L[idx_s, j + 1],
                            L[idx, j + 1]
                        )
                        L[idx_s, j] = self._minsum_f(
                            R[idx, j + 1],
                            L[idx, j + 1]
                        ) + L[idx_s, j + 1]

            # Left-to-right: update R messages at columns 1 to n
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        idx_s = block + i + s
                        R[idx, j] = self._minsum_f(
                            R[idx_s, j - 1] + L[idx_s, j],
                            R[idx, j - 1]
                        )
                        R[idx_s, j] = self._minsum_f(
                            R[idx, j - 1],
                            L[idx, j]
                        ) + R[idx_s, j - 1]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
