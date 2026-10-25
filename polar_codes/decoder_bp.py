"""
极化码 BP（置信传播）译码器
在极化码线性校验矩阵上进行 min-sum BP，含早停
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, polar_generator_matrix


def _gf2_inverse(G):
    """GF(2) 矩阵求逆"""
    N = G.shape[0]
    A = np.concatenate([G.copy(), np.eye(N, dtype=np.int8)], axis=1)
    row = 0
    for col in range(N):
        pivot = None
        for r in range(row, N):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("matrix singular")
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(N):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    return A[:, N:].astype(np.int8)


class BPDecoder:
    """BP 译码器（基于冻结位对应的校验矩阵）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        G = polar_generator_matrix(N)
        self.G_inv = _gf2_inverse(G)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = self.G_inv[frozen_idx, :].astype(np.int8)
        self.M = self.H.shape[0]
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_edges = [np.where(self.H[:, v])[0] for v in range(N)]

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.N, self.M
        Lq = llr_ch.copy()
        Lr = np.zeros((M, N), dtype=np.float64)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                vars_idx = self.cn_edges[m]
                msgs = Lq[vars_idx] - Lr[m, vars_idx]
                for i, v in enumerate(vars_idx):
                    prod = None
                    for j, _ in enumerate(vars_idx):
                        if j == i:
                            continue
                        prod = (
                            msgs[j]
                            if prod is None
                            else self._f_ms(prod, msgs[j])
                        )
                    Lr[m, v] = 0.0 if prod is None else prod

            LQ = np.zeros(N, dtype=np.float64)
            for v in range(N):
                LQ[v] = Lq[v] + np.sum(Lr[self.vn_edges[v], v])

            x_hat = (LQ < 0).astype(int)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        x_hat = (LQ < 0).astype(int)
        u_hat = (x_hat @ self.G_inv) % 2
        return u_hat.astype(int), num_iters
