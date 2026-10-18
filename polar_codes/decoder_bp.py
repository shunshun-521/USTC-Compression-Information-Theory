"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation, _frozen_to_bool
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器（flooding，min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_to_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        i1, i2 = i + j, i + j + step
                        L[stage - 1, i1] = _ms_f(
                            R[stage - 1, i1] + L[stage, i2],
                            L[stage, i1],
                            self.alpha,
                        )
                        L[stage - 1, i2] = _ms_f(
                            R[stage - 1, i1],
                            L[stage, i1],
                            self.alpha,
                        ) + L[stage, i2]

            # 左到右更新 R
            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        i1, i2 = i + j, i + j + step
                        R[stage, i1] = _ms_f(
                            R[stage, i2] + L[stage, i2],
                            R[stage - 1, i1],
                            self.alpha,
                        )
                        R[stage, i2] = _ms_f(
                            R[stage - 1, i1],
                            L[stage, i2],
                            self.alpha,
                        ) + R[stage, i2]

            u_hat = np.zeros(N, dtype=int)
            total = L[0, :] + R[0, :]
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[0, :] + R[0, :]
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
