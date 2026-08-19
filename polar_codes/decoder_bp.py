"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import f_operation as bp_f_base


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            for stage in range(1, n + 1):
                s = stage - 1
                stride = 1 << s
                block = stride << 1
                for base in range(0, N, block):
                    for k in range(stride):
                        i = base + k
                        j = i + stride
                        L[i, s] = alpha * bp_f_base(
                            R[i, s] + L[j, s + 1], L[i, s + 1]
                        )
                        L[j, s] = alpha * bp_f_base(R[i, s], L[i, s + 1]) + L[j, s + 1]

            for stage in range(1, n + 1):
                s = stage - 1
                stride = 1 << s
                block = stride << 1
                for base in range(0, N, block):
                    for k in range(stride):
                        i = base + k
                        j = i + stride
                        R[i, s + 1] = alpha * bp_f_base(
                            R[j, s] + L[j, s + 1], R[i, s]
                        )
                        R[j, s + 1] = alpha * bp_f_base(R[i, s], L[i, s + 1]) + R[j, s]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
