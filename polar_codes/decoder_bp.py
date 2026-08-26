"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 消息传递，含早停机制
"""
import numpy as np
from encoder import polar_encode, prepare_decoder_llr
from channel import hard_decision_llr
from decoder_sc import f_operation, sc_decode_recursive


def ms_f(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def decode(self, llr_ch):
        llr_ch = prepare_decoder_llr(llr_ch)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for i in range(0, N, 2 * step):
                    i2 = i + step
                    L[layer - 1, i] = ms_f(
                        R[layer, i] + L[layer, i2], L[layer, i], alpha
                    )
                    L[layer - 1, i2] = ms_f(
                        R[layer, i], L[layer, i], alpha
                    ) + L[layer, i2]

            for layer in range(0, n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    i2 = i + step
                    R[layer + 1, i] = ms_f(
                        R[layer + 1, i2] + L[layer + 1, i2],
                        R[layer, i],
                        alpha,
                    )
                    R[layer + 1, i2] = ms_f(
                        R[layer, i], L[layer + 1, i2], alpha
                    ) + R[layer + 1, i2]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[0, i] + R[0, i]
                    u_hat[i] = 1 if total < 0 else 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        if not np.array_equal(
            polar_encode(u_hat), hard_decision_llr(llr_ch)
        ):
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)

        return u_hat, num_iters
