"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    L[i, j] / R[i, j]：第 i 个比特节点、第 j 列（0=信源端，n=信道端）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右 → 左更新 L，列 j = n ... 1
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i: i + s, j - 1] = self._f_ms(
                        R[i: i + s, j] + L[i + s: i + 2 * s, j],
                        L[i: i + s, j],
                    )
                    L[i + s: i + 2 * s, j - 1] = (
                        self._f_ms(R[i: i + s, j], L[i: i + s, j])
                        + L[i + s: i + 2 * s, j]
                    )

            # 左 → 右更新 R，列 j = 0 ... n-1
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i: i + s, j + 1] = self._f_ms(
                        R[i + s: i + 2 * s, j] + L[i + s: i + 2 * s, j + 1],
                        R[i: i + s, j],
                    )
                    R[i + s: i + 2 * s, j + 1] = (
                        self._f_ms(R[i: i + s, j], L[i + s: i + 2 * s, j + 1])
                        + R[i + s: i + 2 * s, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
        return u_hat, num_iters
