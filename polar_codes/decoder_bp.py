"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图有 m+1 个 stage（0 到 m），每 stage N 个节点。
  stage 0：信源端；stage m：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.rev = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_dec = self.frozen_bits[self.rev]
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_dec)[0]
        self.info_idx = np.where(~self.frozen_dec)[0]

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, m = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        LARGE = 1e6

        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)

        L[m, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(m, 0, -1):
                block = 1 << (stage - 1)
                for i in range(0, N, 2 * block):
                    for k in range(block):
                        L[stage - 1, i + k] = self._minsum_f(
                            L[stage, i + k + block] + R[stage, i + k],
                            L[stage, i + k],
                        )
                        L[stage - 1, i + k + block] = (
                            self._minsum_f(R[stage, i + k], L[stage, i + k])
                            + L[stage, i + k + block]
                        )

            for stage in range(0, m):
                block = 1 << stage
                for i in range(0, N, 2 * block):
                    for k in range(block):
                        R[stage + 1, i + k] = self._minsum_f(
                            R[stage, i + k + block] + L[stage + 1, i + k + block],
                            R[stage, i + k],
                        )
                        R[stage + 1, i + k + block] = (
                            self._minsum_f(R[stage, i + k], L[stage + 1, i + k])
                            + R[stage, i + k + block]
                        )

            total_llr = L[0, :] + R[0, :]
            u_dec = np.zeros(N, dtype=int)
            u_dec[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
            u_dec[self.frozen_idx] = 0

            u_hat = u_dec[self.rev]
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total_llr = L[0, :] + R[0, :]
            u_dec = np.zeros(N, dtype=int)
            u_dec[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
            u_dec[self.frozen_idx] = 0
            u_hat = u_dec[self.rev]
            u_hat[self.frozen_bits] = 0
            num_iters = self.max_iter

        return u_hat, num_iters
