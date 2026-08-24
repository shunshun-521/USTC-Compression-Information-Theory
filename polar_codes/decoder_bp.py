"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


class BPDecoder:
    """BP 译码器（标准因子图 min-sum BP）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_positions = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f(self, a, b):
        sa = 1.0 if a >= 0 else -1.0
        sb = 1.0 if b >= 0 else -1.0
        return self.alpha * sa * sb * min(abs(a), abs(b))

    def decode(self, llr_ch):
        """返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        for idx in self.frozen_positions:
            R[0, idx] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右→左更新 L（层 n-1 到 0）
            for layer in range(n - 1, -1, -1):
                ni = 1 << layer
                for j in range(0, N, 2 * ni):
                    for k in range(ni):
                        j1 = j + k
                        j2 = j + k + ni
                        L[layer, j1] = self._f(
                            L[layer + 1, j1], L[layer + 1, j2] + R[layer, j1]
                        )
                        L[layer, j2] = L[layer + 1, j2] + self._f(
                            L[layer + 1, j1], R[layer, j1]
                        )

            # 左→右更新 R（层 0 到 n-1）
            for layer in range(0, n):
                ni = 1 << layer
                for j in range(0, N, 2 * ni):
                    for k in range(ni):
                        j1 = j + k
                        j2 = j + k + ni
                        R[layer + 1, j1] = self._f(
                            R[layer, j1], L[layer + 1, j2] + R[layer, j2]
                        )
                        R[layer + 1, j2] = R[layer, j2] + self._f(
                            R[layer, j1], L[layer + 1, j1]
                        )

            for i in range(N):
                if i in self.frozen_positions:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat.astype(int), num_iters
