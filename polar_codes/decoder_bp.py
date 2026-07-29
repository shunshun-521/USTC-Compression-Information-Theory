"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation, prepare_llr_for_decoder
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = set(np.where(self.frozen_bits > 0)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        llr_tree = prepare_llr_for_decoder(llr_ch, N)
        L_msg[:, n] = llr_tree

        for idx in self.frozen_indices:
            R_msg[idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                sp = 1 << (j - 1)
                for block in range(0, N, 2 * sp):
                    for i in range(block, block + sp):
                        L_msg[i, j - 1] = self._f_min_sum(
                            R_msg[i, j] + L_msg[i + sp, j + 1],
                            L_msg[i, j + 1],
                        )
                        L_msg[i + sp, j - 1] = (
                            self._f_min_sum(R_msg[i, j], L_msg[i, j + 1])
                            + L_msg[i + sp, j + 1]
                        )

            for j in range(1, n + 1):
                sp = 1 << (j - 1)
                for block in range(0, N, 2 * sp):
                    for i in range(block, block + sp):
                        R_msg[i, j] = self._f_min_sum(
                            R_msg[i + sp, j] + L_msg[i + sp, j + 1],
                            R_msg[i, j - 1],
                        )
                        R_msg[i + sp, j] = (
                            self._f_min_sum(R_msg[i, j - 1], L_msg[i, j + 1])
                            + R_msg[i + sp, j]
                        )

            for i in range(N):
                total = L_msg[i, 0] + R_msg[i, 0]
                if i in self.frozen_indices:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L_msg[i, 0] + R_msg[i, 0]
            if i in self.frozen_indices:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
