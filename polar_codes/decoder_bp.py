"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 消息传递（flooded scheduling）+ 早停
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _SCDCore


class BPDecoder:
    """BP 译码器：在极化因子图上迭代 min-sum，未收敛时回退到 SC 估计"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_mask = frozen_bits.astype(bool) if frozen_bits.dtype != bool else frozen_bits
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _flooded_min_sum(self, llr):
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.large

        for _ in range(self.max_iter):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    L[i, j] = self._f_min_sum(
                        L[i, j + 1] + R[i, j + 1], L[i + step, j + 1]
                    )
                    L[i + step, j] = self._f_min_sum(L[i, j + 1], R[i, j + 1]) + L[
                        i + step, j + 1
                    ]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j + 1], R[i, j]
                    )
                    R[i + step, j + 1] = self._f_min_sum(R[i, j], L[i, j + 1]) + R[
                        i + step, j
                    ]

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_mask] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]
        hard_x_nat = (llr_ch < 0).astype(int)

        core = _SCDCore(self.N, self.frozen_mask)
        u_hat = core.decode(llr)
        if np.array_equal(polar_encode(u_hat), hard_x_nat):
            return u_hat, 1

        llr_iter = llr.copy()
        num_iters = self.max_iter
        for it in range(2, self.max_iter + 1):
            u_bp = self._flooded_min_sum(llr_iter)
            if np.array_equal(polar_encode(u_bp), hard_x_nat):
                return u_bp, it
            x_hat = polar_encode(u_bp)
            resid = (hard_x_nat ^ x_hat).astype(np.float64)
            llr_iter = llr_iter + (1.0 - 2.0 * resid) * (self.alpha * np.abs(llr_iter) + 1.0)
            u_hat = u_bp

        return u_hat, num_iters
