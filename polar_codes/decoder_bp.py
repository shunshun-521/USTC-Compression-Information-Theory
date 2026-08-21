"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _frozen_mask

_LARGE = 1e6


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        br = self.br

        # 信道 LLR 倒序后与 SC 一致
        ch = llr_ch[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = _LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        La = R[idx, j - 1] + L[idx + s, j]
                        Lb = L[idx, j]
                        L[idx, j - 1] = self._ms(La, Lb)
                        L[idx + s, j - 1] = self._ms(R[idx, j - 1], L[idx, j]) + L[idx + s, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = self._ms(
                            R[idx + s, j - 1] + L[idx + s, j], R[idx, j - 1]
                        )
                        R[idx + s, j] = self._ms(R[idx, j - 1], L[idx, j]) + R[idx + s, j - 1]

            # 判决与早停
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(self.frozen, 0, (total < 0).astype(int))
            x_hat = polar_encode(u_hat)
            hard_ch = (ch < 0).astype(int)
            x_hard = np.zeros(N, dtype=int)
            x_hard[br] = hard_ch
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(self.frozen, 0, (total < 0).astype(int))
        return u_hat.astype(int), num_iters
