"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, LLR_CLIP

LARGE = 1e6


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.brp = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = np.clip(llr_ch[self.brp], -LLR_CLIP, LLR_CLIP)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for block in range(0, N, step << 1):
                    for j in range(step):
                        i0 = block + j
                        i1 = block + j + step
                        L[i0, layer] = self._f(
                            R[i0, layer + 1] + L[i1, layer + 1],
                            L[i0, layer + 1],
                        )
                        L[i1, layer] = self._f(
                            R[i0, layer + 1], L[i0, layer + 1]
                        ) + L[i1, layer + 1]

            for layer in range(n):
                step = 1 << layer
                for block in range(0, N, step << 1):
                    for j in range(step):
                        i0 = block + j
                        i1 = block + j + step
                        R[i0, layer + 1] = self._f(
                            R[i1, layer] + L[i1, layer + 1], R[i0, layer]
                        )
                        R[i1, layer + 1] = self._f(
                            R[i0, layer], L[i0, layer + 1]
                        ) + R[i1, layer]

            R[self.frozen_idx, 0] = LARGE
            num_iters = it
            u_hat = self._hard_decision(L, R)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._hard_decision(L, R)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        return (total < 0).astype(np.int8)
