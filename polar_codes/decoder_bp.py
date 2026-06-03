"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 近似；采用迭代软消息增强 + SC 判决，
在极化因子图上实现 flooded BP 的实用近似，并支持早停。
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode, f_operation


def _minsum_f(a, b, alpha=1.0):
    return alpha * f_operation(a, b)


class BPDecoder:
    """
    BP 译码器。

    实现分层 min-sum 消息传递；若未收敛，则回退到迭代增强 SC（IRE-SC）
    以保证与极化编码一致的有效译码输出。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool).flatten()
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _layered_bp_pass(self, L, R, llr_ch):
        """单次分层 BP 更新"""
        n, N, alpha = self.n, self.N, self.alpha

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                L[i, j - 1] = _minsum_f(
                    R[i, j] + L[i + s, j], L[i, j], alpha
                )
                L[i + s, j - 1] = _minsum_f(
                    R[i, j], L[i, j], alpha
                ) + L[i + s, j]

        for j in range(1, n + 1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                R[i, j] = _minsum_f(
                    R[i + s, j] + L[i + s, j], R[i, j - 1], alpha
                )
                R[i + s, j] = _minsum_f(
                    R[i, j - 1], L[i, j], alpha
                ) + R[i + s, j]

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).flatten()
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1
            u_hat = self._layered_bp_pass(L, R, llr_ch)

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                return u_hat, num_iters

        # 迭代增强 SC 回退（IRE-SC）
        llr_work = llr_ch.copy()
        for it in range(self.max_iter):
            num_iters = self.max_iter + it + 1
            u_hat = sc_decode(llr_work, self.frozen_bits)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break
            llr_work = llr_ch + 0.75 * (1.0 - 2.0 * x_hat)

        return u_hat, num_iters
