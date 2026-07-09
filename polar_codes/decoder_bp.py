"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        self.rev = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        n = self.n
        N = self.N
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        channel = llr_nat[self.rev]

        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        L[n][:] = channel
        R[0][:] = 0.0
        R[0][self.frozen_idx] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    L[layer][i:i + step] = self._f_ms(
                        L[layer + 1][i:i + step],
                        L[layer + 1][i + step:i + 2 * step]
                    )
                    L[layer][i + step:i + 2 * step] = (
                        self._f_ms(L[layer + 1][i:i + step], R[layer][i:i + step])
                        + L[layer + 1][i + step:i + 2 * step]
                    )

            for layer in range(0, n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    R[layer + 1][i:i + step] = self._f_ms(
                        R[layer][i:i + step] + L[layer + 1][i + step:i + 2 * step],
                        R[layer][i + step:i + 2 * step]
                    )
                    R[layer + 1][i + step:i + 2 * step] = (
                        self._f_ms(R[layer][i:i + step], L[layer + 1][i:i + step])
                        + R[layer][i + step:i + 2 * step]
                    )

            for i in range(N):
                total = L[0][i] + R[0][i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[0][i] + R[0][i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
