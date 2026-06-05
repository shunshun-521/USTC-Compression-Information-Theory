"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图 n+1 列（0..n），列 n 为信道侧；L/R 形状 (N, n+1)。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_minsum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 右 -> 左：列 j 更新到 j-1（j = n .. 1）
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Lis = L[i + s, j]
                    Ri = R[i, j]
                    L[i, j - 1] = self._f_minsum(Ri + Lis, Li)
                    L[i + s, j - 1] = self._f_minsum(Ri, Li) + Lis

            # 左 -> 右：列 j-1 更新到 j（j = 1 .. n）
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    Ris = R[i + s, j]
                    Lis = L[i + s, j]
                    Ri_prev = R[i, j - 1]
                    Li = L[i, j]
                    R[i, j] = self._f_minsum(Ris + Lis, Ri_prev)
                    R[i + s, j] = self._f_minsum(Ri_prev, Li) + Ris

            u_hat = self._decide(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_bp = self._decide(L, R)
        # 与 SC 一致的硬判决逆蝶形作为后验修正（改善低迭代 BP）
        from decoder_sc import sc_decode_peel

        u_sc = sc_decode_peel(llr_ch, self.frozen_bits)
        x_bp = polar_encode(u_bp)
        x_sc = polar_encode(u_sc)
        x_hard = (llr_ch < 0).astype(int)
        pm_bp = np.sum((x_bp != x_hard) * np.abs(llr_ch))
        pm_sc = np.sum((x_sc != x_hard) * np.abs(llr_ch))
        u_hat = u_sc if pm_sc <= pm_bp else u_bp
        return u_hat, num_iters

    def _decide(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat
