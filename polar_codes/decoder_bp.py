"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _channel_llr_to_decoder
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = _channel_llr_to_decoder(llr_ch, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                stride = 2 ** (stage + 1)
                half = stride // 2
                for block in range(0, N, stride):
                    for j in range(half):
                        idx = block + j
                        R[idx, stage + 1] = _f_min_sum(
                            R[idx, stage],
                            L[idx, stage + 1] + R[idx + half, stage],
                            self.alpha,
                        )
                        R[idx + half, stage + 1] = _f_min_sum(
                            R[idx, stage],
                            L[idx, stage + 1],
                            self.alpha,
                        ) + R[idx + half, stage]

            for stage in range(n - 1, -1, -1):
                stride = 2 ** (stage + 1)
                half = stride // 2
                for block in range(0, N, stride):
                    for j in range(half):
                        idx = block + j
                        L[idx, stage] = _f_min_sum(
                            L[idx, stage + 1],
                            L[idx + half, stage + 1] + R[idx + half, stage],
                            self.alpha,
                        )
                        L[idx + half, stage] = _f_min_sum(
                            R[idx, stage],
                            L[idx, stage + 1],
                            self.alpha,
                        ) + L[idx + half, stage + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
