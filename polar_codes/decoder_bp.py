"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha=0.9375):
    """Min-sum f 运算（带缩放因子）。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev]
        N = self.N
        m = self.n
        alpha = self.alpha

        # L[i][λ]: 右向左消息, R[i][λ]: 左向右消息, λ=0..m
        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            # 右向左更新 L
            for lam in range(m, 0, -1):
                stride = 1 << (lam - 1)
                for phi in range(0, N, 2 * stride):
                    for omega in range(phi, phi + stride):
                        L[omega, lam - 1] = _minsum_f(
                            R[omega, lam] + L[omega + stride, lam], L[omega, lam], alpha
                        )
                        L[omega + stride, lam - 1] = (
                            _minsum_f(R[omega, lam], L[omega, lam], alpha) + L[omega + stride, lam]
                        )

            # 左向右更新 R
            for lam in range(1, m + 1):
                stride = 1 << (lam - 1)
                for phi in range(0, N, 2 * stride):
                    for omega in range(phi, phi + stride):
                        R[omega, lam] = _minsum_f(
                            R[omega + stride, lam] + L[omega + stride, lam], R[omega, lam - 1], alpha
                        )
                        R[omega + stride, lam] = (
                            _minsum_f(R[omega, lam - 1], L[omega, lam], alpha) + R[omega + stride, lam]
                        )

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
