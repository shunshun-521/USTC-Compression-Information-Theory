"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, prepare_channel_llr
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.inv_br = np.empty(N, dtype=int)
        self.inv_br[self.br] = np.arange(N)
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        llr_ch: 自然顺序信道 LLR。
        """
        llr_ch = prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in np.where(self.frozen_bits)[0]:
            R[idx, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        i2 = i + s
                        L[i, j - 1] = self._f_ms(
                            R[i, j] + L[i2, j], L[i, j]
                        )
                        L[i2, j - 1] = self._f_ms(
                            R[i, j], L[i, j]
                        ) + L[i2, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        i2 = i + s
                        R[i, j] = self._f_ms(
                            R[i2, j] + L[i2, j], R[i, j - 1]
                        )
                        R[i2, j] = self._f_ms(
                            R[i, j - 1], L[i, j]
                        ) + R[i2, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

        return u_hat, num_iters
