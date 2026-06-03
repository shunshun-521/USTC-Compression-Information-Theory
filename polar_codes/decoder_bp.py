"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, ml_polar_decode


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, alpha = self.n, self.N, self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = 1e8

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        i = block + j
                        i2 = i + step
                        L[stage - 1, i] = _minsum_f(
                            R[stage, i] + L[stage, i2], L[stage, i], alpha
                        )
                        L[stage - 1, i2] = _minsum_f(
                            R[stage, i], L[stage, i], alpha
                        ) + L[stage, i2]

            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        i = block + j
                        i2 = i + step
                        R[stage, i] = _minsum_f(
                            R[stage, i2] + L[stage, i2], R[stage - 1, i], alpha
                        )
                        R[stage, i2] = _minsum_f(
                            R[stage - 1, i], L[stage, i], alpha
                        ) + R[stage, i2]

            u_hat = (L[0, :] + R[0, :] >= 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break

        u_hat = (L[0, :] + R[0, :] >= 0).astype(int)
        u_hat[self.frozen_bits] = 0
        x_hd = (llr_ch < 0).astype(int)
        if np.sum(polar_encode(u_hat) != x_hd) > 0:
            u_hat = ml_polar_decode(llr_ch, self.frozen_bits)
        return u_hat, num_iters
