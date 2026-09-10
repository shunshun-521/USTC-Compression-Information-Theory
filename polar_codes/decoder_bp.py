"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, remap_channel_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _ms_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, La, Ra):
        total = La[0, :] + Ra[0, :]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat

    def decode(self, llr_ch):
        llr_ch = remap_channel_llr(llr_ch)
        n = self.n
        N = self.N

        La = np.zeros((n + 1, N), dtype=np.float64)
        Ra = np.zeros((n + 1, N), dtype=np.float64)
        La[n, :] = np.clip(llr_ch, -30.0, 30.0)
        Ra[0, :] = 0.0
        Ra[0, self.frozen_bits] = self.LARGE

        best_u = self._hard_decision(La, Ra)
        num_iters = 1

        for it in range(1, self.max_iter + 1):
            La_old = La.copy()
            Ra_old = Ra.copy()

            for layer in range(n, 0, -1):
                block = 1 << (layer - 1)
                for phi in range(0, N, 2 * block):
                    for beta in range(block):
                        psi = phi + beta
                        La[layer - 1, psi] = self._ms_f(
                            La[layer, psi] + Ra[layer - 1, psi],
                            La[layer, psi + block],
                        )
                        La[layer - 1, psi + block] = (
                            self._ms_f(Ra[layer - 1, psi], La[layer, psi])
                            + La[layer, psi + block]
                        )

            for layer in range(1, n + 1):
                block = 1 << (layer - 1)
                for phi in range(0, N, 2 * block):
                    for beta in range(block):
                        psi = phi + beta
                        Ra[layer, psi + block] = self._ms_f(
                            Ra[layer - 1, psi + block] + La[layer, psi + block],
                            Ra[layer - 1, psi],
                        )
                        Ra[layer, psi] = (
                            self._ms_f(Ra[layer - 1, psi], La[layer, psi + block])
                            + Ra[layer - 1, psi + block]
                        )

            if it > 1:
                La = 0.5 * La + 0.5 * La_old
                Ra = 0.5 * Ra + 0.5 * Ra_old

            u_hat = self._hard_decision(La, Ra)
            if it == 1:
                best_u = u_hat.copy()

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                return u_hat, num_iters

        return best_u, num_iters
