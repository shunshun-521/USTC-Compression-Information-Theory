"""
极化码 BP（置信传播）译码器
基于因子图 min-sum BP（MDPI 2022 公式），含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _g_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器：在极化因子图上执行迭代 min-sum 消息传递。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        n = self.n
        N = self.N
        LARGE = 1e7
        rho = 0.9

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L_old = L.copy()
            R_old = R.copy()

            for stage in range(n - 1, -1, -1):
                s = 1 << stage
                for block in range(0, N, 2 * s):
                    for j in range(s):
                        i0 = block + j
                        i1 = block + j + s
                        L[i0, stage] = (
                            rho
                            * _g_min_sum(
                                L_old[i0, stage + 1],
                                L_old[i1, stage + 1] + R_old[i1, stage],
                                self.alpha,
                            )
                            + (1 - rho) * L_old[i0, stage]
                        )
                        L[i1, stage] = (
                            rho
                            * (
                                _g_min_sum(L_old[i0, stage + 1], R_old[i0, stage], self.alpha)
                                + L_old[i1, stage + 1]
                            )
                            + (1 - rho) * L_old[i1, stage]
                        )

            for stage in range(n):
                s = 1 << stage
                for block in range(0, N, 2 * s):
                    for j in range(s):
                        i0 = block + j
                        i1 = block + j + s
                        R[i0, stage + 1] = (
                            rho
                            * _g_min_sum(
                                R_old[i0, stage],
                                L[i1, stage + 1] + R_old[i1, stage],
                                self.alpha,
                            )
                            + (1 - rho) * R_old[i0, stage + 1]
                        )
                        R[i1, stage + 1] = (
                            rho
                            * (
                                _g_min_sum(L[i0, stage + 1], R_old[i0, stage], self.alpha)
                                + R_old[i1, stage]
                            )
                            + (1 - rho) * R_old[i1, stage + 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
