"""
极化码 BP（置信传播）译码器
基于极化码校验矩阵 Tanner 图，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import build_generator_matrix, gf2_inverse


class BPDecoder:
    """BP 译码器（Tanner 图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.875):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        G = build_generator_matrix(N)
        Ginv = gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = Ginv[frozen_idx, :]
        self.Ginv = Ginv

        self._chk_edges = []
        for m in range(self.H.shape[0]):
            self._chk_edges.append(np.where(self.H[m])[0])

    def _check_update(self, Lq, R, m, idx, alpha):
        qvals = Lq[idx] - R[m, idx]
        d = len(idx)
        if d == 1:
            R[m, idx[0]] = 0.0
            return
        abs_q = np.abs(qvals)
        signs = np.sign(qvals)
        signs[signs == 0] = 1.0
        prod_sign = np.prod(signs)
        for k in range(d):
            sign_excl = prod_sign * signs[k]
            if d == 2:
                min_excl = abs_q[1 - k]
            else:
                min_excl = np.min(abs_q[np.arange(d) != k])
            R[m, idx[k]] = alpha * sign_excl * min_excl

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        M, N = self.H.shape
        R = np.zeros((M, N), dtype=np.float64)
        Lq = llr_ch.copy()
        alpha = self.alpha
        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for m, idx in enumerate(self._chk_edges):
                if len(idx) >= 2:
                    self._check_update(Lq, R, m, idx, alpha)

            Lq = llr_ch + np.sum(R, axis=0)

            if it % 2 == 1 or it == self.max_iter - 1:
                x_hat_bits = (Lq < 0).astype(int)
                if np.all((self.H @ x_hat_bits) % 2 == 0):
                    u_hat = (x_hat_bits @ self.Ginv) % 2
                    u_hat[self.frozen_bits] = 0
                    break

        if num_iters == self.max_iter or not np.any(u_hat):
            x_hat_bits = (Lq < 0).astype(int)
            u_hat = (x_hat_bits @ self.Ginv) % 2
            u_hat[self.frozen_bits] = 0

        return u_hat.astype(int), num_iters
