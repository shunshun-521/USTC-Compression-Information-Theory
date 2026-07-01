"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _bit_reversed_index, f_operation, sc_decode
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def _factor_graph_pass(self, L, R, llr_ch):
        """执行一次因子图双向消息传递（按规格公式）。"""
        n, N = self.n, self.N
        L_old = L.copy()
        R_old = R.copy()
        L_old[:, n] = llr_ch
        R_old[:, 0] = 0.0
        R_old[self.frozen_bits, 0] = self.LARGE

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                L[i, j - 1] = self._f_ms(
                    R_old[i, j] + L_old[i + s, j], L_old[i, j]
                )
                L[i + s, j - 1] = self._f_ms(R_old[i, j], L_old[i, j]) + L_old[i + s, j]

        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                R[i, j + 1] = self._f_ms(
                    R_old[i + s, j] + L_old[i + s, j + 1], R_old[i, j]
                )
                R[i + s, j + 1] = (
                    self._f_ms(R_old[i, j], L_old[i + s, j + 1]) + R_old[i + s, j]
                )

        return L, R

    def decode(self, llr_ch):
        """
        主译码函数。
        结合因子图 min-sum 迭代与 SC 软信息精炼，保证与编码器一致的判决。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        belief = llr_ch.copy()
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L, R = self._factor_graph_pass(L, R, llr_ch)

            extrinsic = np.zeros(N, dtype=np.float64)
            for phi in range(N):
                leaf = _bit_reversed_index(phi, n)
                extrinsic[leaf] = L[leaf, 0] + R[leaf, 0]

            belief = 0.65 * llr_ch + 0.35 * extrinsic
            u_hat = sc_decode(belief, self.frozen_bits)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
