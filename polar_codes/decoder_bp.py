"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_g(x, y, alpha):
    """min-sum 校验节点运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        br = bit_reversal_permutation(N)
        frozen_br = self.frozen_bits[br]
        R[0, :] = 0.0
        R[0, frozen_br] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                stride = 1 << s
                block = stride << 1
                for base in range(0, N, block):
                    for j in range(base, base + stride):
                        jp = j + stride
                        L[s, j] = _ms_g(
                            L[s + 1, j],
                            L[s + 1, jp] + R[s, jp],
                            self.alpha,
                        )
                        L[s, jp] = (
                            _ms_g(L[s + 1, j], R[s, j], self.alpha)
                            + L[s + 1, jp]
                        )

            for s in range(n):
                stride = 1 << s
                block = stride << 1
                for base in range(0, N, block):
                    for j in range(base, base + stride):
                        jp = j + stride
                        R[s + 1, j] = _ms_g(
                            R[s, j],
                            L[s + 1, jp] + R[s, jp],
                            self.alpha,
                        )
                        R[s + 1, jp] = (
                            _ms_g(L[s + 1, j], R[s, j], self.alpha) + R[s, jp]
                        )

            num_iters = it
            total = L[0, :] + R[0, :]
            u_internal = (total < 0).astype(int)
            u_internal[frozen_br] = 0
            u_hat = u_internal[br]

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_internal = (total < 0).astype(int)
        u_internal[frozen_br] = 0
        u_hat = u_internal[br]
        return u_hat, num_iters
