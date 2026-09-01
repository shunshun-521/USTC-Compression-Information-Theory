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
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e9

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for phi in range(n - 1, -1, -1):
                step = 1 << phi
                for beta in range(0, N, step * 2):
                    for omega in range(beta, beta + step):
                        L[omega, phi] = self._f_min_sum(
                            R[omega, phi] + L[omega + step, phi + 1],
                            L[omega, phi + 1],
                        )
                        L[omega + step, phi] = self._f_min_sum(
                            R[omega, phi],
                            L[omega, phi + 1],
                        ) + L[omega + step, phi + 1]

            for phi in range(n):
                step = 1 << phi
                for beta in range(0, N, step * 2):
                    for omega in range(beta, beta + step):
                        R[omega, phi + 1] = self._f_min_sum(
                            R[omega + step, phi] + L[omega + step, phi + 1],
                            R[omega, phi],
                        )
                        R[omega + step, phi + 1] = self._f_min_sum(
                            R[omega, phi],
                            L[omega, phi + 1],
                        ) + R[omega + step, phi]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
