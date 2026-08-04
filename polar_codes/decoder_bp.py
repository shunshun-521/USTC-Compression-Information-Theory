"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, prepare_llr_for_decode


class BPDecoder:
    """
    BP 译码器。
    因子图 n+1 列（0..n），列 0 为信源端，列 n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_dec = prepare_llr_for_decode(llr_ch, N)

        # L: 左向消息 [node, stage], R: 右向消息 [node, stage]
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_dec
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for stage in range(n, 0, -1):
                stride = 2 ** (stage - 1)
                for block in range(0, N, 2 * stride):
                    for j in range(stride):
                        i = block + j
                        ip = i + stride
                        L[i, stage - 1] = self._f_min_sum(
                            R[i, stage - 1] + L[ip, stage],
                            L[i, stage],
                        )
                        L[ip, stage - 1] = (
                            self._f_min_sum(R[i, stage - 1], L[i, stage])
                            + L[ip, stage]
                        )

            # 左到右更新 R
            for stage in range(1, n + 1):
                stride = 2 ** (stage - 1)
                for block in range(0, N, 2 * stride):
                    for j in range(stride):
                        i = block + j
                        ip = i + stride
                        R[i, stage] = self._f_min_sum(
                            R[ip, stage] + L[ip, stage],
                            R[i, stage - 1],
                        )
                        R[ip, stage] = (
                            self._f_min_sum(R[i, stage - 1], L[ip, stage])
                            + R[ip, stage]
                        )

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
