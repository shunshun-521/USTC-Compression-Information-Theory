"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _ms_boxplus(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        top, bottom = i + t, i + t + step
                        L[top, j - 1] = self._ms_boxplus(
                            R[top, j] + L[bottom, j], L[top, j]
                        )
                        L[bottom, j - 1] = self._ms_boxplus(
                            R[top, j], L[top, j]
                        ) + L[bottom, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        top, bottom = i + t, i + t + step
                        R[top, j + 1] = self._ms_boxplus(
                            R[bottom, j] + L[bottom, j + 1], R[top, j]
                        )
                        R[bottom, j + 1] = self._ms_boxplus(
                            R[top, j], L[top, j + 1]
                        ) + R[bottom, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr

    N, K = 16, 8
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    u = np.zeros(N, dtype=int)
    u[info_idx] = [1, 0, 1, 0, 1, 1, 0, 1]
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.05)
    u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
    print("BP noiseless-ish:", np.array_equal(u_hat[info_idx], u[info_idx]), "iters", iters)
