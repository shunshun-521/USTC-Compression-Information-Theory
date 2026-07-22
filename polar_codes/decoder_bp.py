"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decode(self, L_msg, R_msg):
        u_hat = np.zeros(self.N, dtype=int)
        total = L_msg[0, :] + R_msg[0, :]
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.rev]
        N, n = self.N, self.n

        L_msg = np.zeros((n + 1, N), dtype=np.float64)
        R_msg = np.zeros((n + 1, N), dtype=np.float64)
        L_msg[n, :] = llr_internal
        R_msg[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        a = block + i
                        b = block + i + step
                        L_msg[stage - 1, a] = self._f_min_sum(
                            R_msg[stage, a] + L_msg[stage, b],
                            L_msg[stage, a],
                        )
                        L_msg[stage - 1, b] = (
                            self._f_min_sum(R_msg[stage, a], L_msg[stage, a])
                            + L_msg[stage, b]
                        )

            for stage in range(0, n):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        a = block + i
                        b = block + i + step
                        R_msg[stage + 1, a] = self._f_min_sum(
                            R_msg[stage, b] + L_msg[stage + 1, b],
                            R_msg[stage, a],
                        )
                        R_msg[stage + 1, b] = (
                            self._f_min_sum(R_msg[stage, a], L_msg[stage + 1, a])
                            + R_msg[stage, b]
                        )

            u_hat = self._hard_decode(L_msg, R_msg)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        u_hat = self._hard_decode(L_msg, R_msg)
        return u_hat, num_iters
