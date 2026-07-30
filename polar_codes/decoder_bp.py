"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _permute_channel_llrs


def _minsum_f(a, b, alpha):
    """min-sum f 运算。"""
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen_bits = frozen_bits
        else:
            self.frozen_bits = frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = _permute_channel_llrs(llr_ch)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        u = i + k
                        v = i + k + step
                        L[u, layer] = _minsum_f(
                            R[u, layer] + L[v, layer + 1], L[u, layer + 1], alpha
                        )
                        L[v, layer] = _minsum_f(
                            R[u, layer], L[u, layer + 1], alpha
                        ) + L[v, layer + 1]

            for layer in range(n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        u = i + k
                        v = i + k + step
                        R[u, layer + 1] = _minsum_f(
                            R[v, layer] + L[v, layer + 1], R[u, layer], alpha
                        )
                        R[v, layer + 1] = _minsum_f(
                            R[u, layer], L[u, layer + 1], alpha
                        ) + R[v, layer]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            inv_br = np.argsort(br)
            hard_ch = (llr_ch[inv_br] < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
