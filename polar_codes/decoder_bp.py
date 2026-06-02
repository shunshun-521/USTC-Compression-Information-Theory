"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器（按极化码因子图 stages 更新 L/R 消息）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        # L[i, s] 与 R[i, s]：stage s=0 为信源侧，s=n 为信道侧
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（stage n-1 ... 0）
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        L[a, s] = _minsum_f(R[a, s] + L[b, s + 1], L[a, s + 1], alpha)
                        L[b, s] = _minsum_f(R[a, s], L[a, s + 1], alpha) + L[b, s + 1]

            # 左到右更新 R（stage 0 ... n-1）
            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        R[a, s + 1] = _minsum_f(
                            R[b, s] + L[b, s + 1], R[a, s], alpha
                        )
                        R[b, s + 1] = _minsum_f(R[a, s], L[a, s + 1], alpha) + R[b, s]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
