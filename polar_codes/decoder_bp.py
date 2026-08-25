"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _ms_f(a, b, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha

        # L[i][j]: 从右到左消息，列 j=0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        for i in range(N):
            L[i, n] = llr_ch[self.br[i]]

        R[:, 0] = 0.0
        for idx in self.frozen_idx:
            R[idx, 0] = self.large

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = _ms_f(
                            R[i, j] + L[i + s, j],
                            L[i, j],
                            alpha,
                        )
                        L[i + s, j - 1] = _ms_f(
                            R[i, j],
                            L[i, j],
                            alpha,
                        ) + L[i + s, j]

            # 从左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j] = _ms_f(
                            R[i + s, j] + L[i + s, j],
                            R[i, j - 1],
                            alpha,
                        )
                        R[i + s, j] = _ms_f(
                            R[i, j - 1],
                            L[i, j],
                            alpha,
                        ) + R[i + s, j]

            # 早停检查
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
