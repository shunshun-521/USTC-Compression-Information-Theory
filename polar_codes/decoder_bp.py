"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 Tanner 图，使用 min-sum 近似，含早停
"""
import numpy as np
from encoder import polar_encode, build_generator_matrix


def _gf2_inverse(G):
    """GF(2) 矩阵求逆"""
    n = G.shape[0]
    aug = np.hstack([G.copy(), np.eye(n, dtype=int)])
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2)")
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    return aug[:, n:]


class BPDecoder:
    """
    BP 译码器（基于极化码校验矩阵）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

        G = build_generator_matrix(N)
        self.Ginv = _gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.H = self.Ginv[frozen_idx, :]
        self.M = self.H.shape[0]
        self.var_to_chk = [np.where(self.H[:, i])[0] for i in range(N)]
        self.chk_to_var = [np.where(self.H[m, :])[0] for m in range(self.M)]

    def _chk_msg(self, vals):
        if len(vals) == 0:
            return 0.0
        signs = np.prod(np.sign(vals)) if np.all(vals != 0) else 1.0
        return self.alpha * signs * np.min(np.abs(vals))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.M

        Lq = np.zeros((M, N), dtype=np.float64)
        Lr = np.zeros((M, N), dtype=np.float64)
        Lv = llr_ch.copy()

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                vars_m = self.chk_to_var[m]
                for i in vars_m:
                    others = [j for j in vars_m if j != i]
                    incoming = [Lv[j] - Lr[m, j] for j in others]
                    Lq[m, i] = self._chk_msg(np.array(incoming))

            posterior = np.zeros(N, dtype=np.float64)
            for i in range(N):
                posterior[i] = Lv[i] + np.sum(Lq[self.var_to_chk[i], i])

            for i in range(N):
                for m in self.var_to_chk[i]:
                    Lr[m, i] = posterior[i] - Lq[m, i]

            x_hat = (posterior < 0).astype(int)
            u_hat = (self.Ginv @ x_hat) % 2
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_reenc = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_reenc, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
