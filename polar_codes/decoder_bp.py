"""
极化码 BP（置信传播）译码器
基于校验矩阵 H = G^{-1}[frozen,:]，min-sum 近似，含早停
"""
import numpy as np
from encoder import polar_encode, build_generator_matrix
from channel import hard_decision_llr


def _gf2_inverse(G):
    N = G.shape[0]
    A = G.copy().astype(np.int8)
    I = np.eye(N, dtype=np.int8)
    for col in range(N):
        if A[col, col] == 0:
            for r in range(col + 1, N):
                if A[r, col] == 1:
                    A[[col, r]] = A[[r, col]]
                    I[[col, r]] = I[[r, col]]
                    break
        if A[col, col] == 0:
            raise ValueError("Singular matrix")
        for r in range(N):
            if r != col and A[r, col] == 1:
                A[r] = (A[r] + A[col]) % 2
                I[r] = (I[r] + I[col]) % 2
    return I


class BPDecoder:
    """BP 译码器（校验矩阵 Tanner 图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        G = build_generator_matrix(N)
        Ginv = _gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = Ginv[frozen_idx, :].astype(np.int8)
        self.M = self.H.shape[0]
        self._cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self._vn_edges = [np.where(self.H[:, n])[0] for n in range(N)]
        self._Ginv = Ginv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.N, self.M

        Lq = llr_ch.copy()
        Rcv = {(m, n): 0.0 for m in range(M) for n in self._cn_edges[m]}

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            Rcv_new = {}
            for m in range(M):
                nodes = self._cn_edges[m]
                msgs = [Lq[n] - Rcv[(m, n)] for n in nodes]
                for idx, n in enumerate(nodes):
                    other = [msgs[j] for j in range(len(nodes)) if j != idx]
                    prod_sign = 1.0
                    min_abs = np.inf
                    for v in other:
                        prod_sign *= 1.0 if v >= 0 else -1.0
                        min_abs = min(min_abs, abs(v))
                    Rcv_new[(m, n)] = self.alpha * prod_sign * min_abs

            Rcv = Rcv_new
            for n in range(N):
                Lq[n] = llr_ch[n] + sum(Rcv[(m, n)] for m in self._vn_edges[n])

            x_hard = hard_decision_llr(Lq)
            u_hat = (self._Ginv @ x_hard.astype(np.int8)) % 2
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        x_hard = hard_decision_llr(Lq)
        u_hat = (self._Ginv @ x_hard.astype(np.int8)) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
