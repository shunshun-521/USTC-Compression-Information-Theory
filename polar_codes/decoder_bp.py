"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _boxplus(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _boxplus_g(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


class BPDecoder:
    """BP 译码器（Permuted 因子图，与 SC 使用相同的 LLR 倒序）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_log_domain=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.br = bit_reversal_permutation(N)
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_log_domain = use_log_domain
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _f_bp(self, a, b):
        if self.use_log_domain:
            return _boxplus(a, b)
        return self._f_min_sum(a, b)

    def _g_bp(self, a, b, u_bit):
        if self.use_log_domain:
            return _boxplus_g(a, b, u_bit)
        return (1.0 - 2.0 * u_bit) * a + b

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        br = self.br

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            # 右 -> 左（列 j: n .. 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx0 = i + t
                        idx1 = i + t + s
                        L[idx0, j - 1] = self._f_bp(
                            R[idx0, j - 1] + L[idx1, j], L[idx0, j]
                        )
                        L[idx1, j - 1] = self._f_bp(R[idx0, j - 1], L[idx0, j]) + L[idx1, j]

            # 左 -> 右（列 j: 0 .. n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx0 = i + t
                        idx1 = i + t + s
                        R[idx0, j + 1] = self._f_bp(
                            R[idx1, j] + L[idx1, j + 1], R[idx0, j]
                        )
                        R[idx1, j + 1] = self._f_bp(R[idx0, j], L[idx0, j + 1]) + R[idx1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
