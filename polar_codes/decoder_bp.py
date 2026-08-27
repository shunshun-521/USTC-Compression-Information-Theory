"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for step in range(n - 1, -1, -1):
                block = 1 << step
                for i in range(0, N, 2 * block):
                    for j in range(i, i + block):
                        L[j, step] = _ms_f(
                            R[j, step + 1] + L[j + block, step + 1],
                            L[j, step + 1],
                            self.alpha,
                        )
                        L[j + block, step] = (
                            _ms_f(R[j, step + 1], L[j, step + 1], self.alpha)
                            + L[j + block, step + 1]
                        )

            for step in range(1, n + 1):
                block = 1 << (step - 1)
                for i in range(0, N, 2 * block):
                    for j in range(i, i + block):
                        R[j, step] = _ms_f(
                            R[j + block, step] + L[j + block, step],
                            R[j, step - 1],
                            self.alpha,
                        )
                        R[j + block, step] = (
                            _ms_f(R[j, step - 1], L[j, step], self.alpha)
                            + R[j + block, step - 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_indices] = 0

        return u_hat, num_iters
