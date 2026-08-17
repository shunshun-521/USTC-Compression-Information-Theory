"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation


def _build_generator_matrix(N):
    """构造极化码生成矩阵 G_N = B_N F^{⊗n}。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    n = int(math.log2(N))
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, rev[i]] = 1
    return G @ B % 2


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.G = _build_generator_matrix(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数。"""
        N = self.N
        m = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = 1e10

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(m - 1, -1, -1):
                span = 2 ** j
                for i in range(0, N, 2 * span):
                    L[i, j] = self._f_min_sum(
                        L[i, j + 1], L[i + span, j + 1] + R[i + span, j]
                    )
                    L[i + span, j] = self._f_min_sum(
                        R[i, j], L[i, j + 1]
                    ) + L[i + span, j + 1]

            for j in range(m):
                span = 2 ** j
                for i in range(0, N, 2 * span):
                    R[i, j + 1] = self._f_min_sum(
                        R[i, j], L[i + span, j + 1] + R[i + span, j]
                    )
                    R[i + span, j + 1] = self._f_min_sum(
                        R[i, j], L[i, j + 1]
                    ) + R[i + span, j]

            x_llr = L[:, m] + R[:, m]
            x_hat = (x_llr < 0).astype(int)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        x_llr = L[:, m] + R[:, m]
        x_hat = (x_llr < 0).astype(int)
        u_hat = (x_hat @ self.G) % 2
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
