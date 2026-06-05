"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import reorder_llr_for_decode
from encoder import polar_encode_no_br


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N = self.N
        n = self.n
        llr_ch = reorder_llr_for_decode(llr_ch, N)

        Lmsg = np.zeros((n + 1, N), dtype=np.float64)
        Rmsg = np.zeros((n + 1, N), dtype=np.float64)

        Lmsg[n, :] = llr_ch
        Rmsg[0, :] = 0.0
        Rmsg[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        u = block + j
                        v = block + j + step
                        Lmsg[layer, u] = _f_min_sum(
                            Rmsg[layer, u] + Lmsg[layer + 1, v],
                            Lmsg[layer + 1, u],
                            self.alpha,
                        )
                        Lmsg[layer, v] = _f_min_sum(
                            Rmsg[layer, u],
                            Lmsg[layer + 1, u],
                            self.alpha,
                        ) + Lmsg[layer + 1, v]

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        u = block + j
                        v = block + j + step
                        Rmsg[layer + 1, u] = _f_min_sum(
                            Rmsg[layer, v] + Lmsg[layer + 1, v],
                            Rmsg[layer, u],
                            self.alpha,
                        )
                        Rmsg[layer + 1, v] = _f_min_sum(
                            Rmsg[layer, u],
                            Lmsg[layer + 1, u],
                            self.alpha,
                        ) + Rmsg[layer, v]

            num_iters = it
            posterior = Lmsg[0, :] + Rmsg[0, :]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if posterior[i] >= 0 else 1

            x_hat = polar_encode_no_br(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        posterior = Lmsg[0, :] + Rmsg[0, :]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if posterior[i] >= 0 else 1

        return u_hat, num_iters
