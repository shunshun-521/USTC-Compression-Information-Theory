"""
极化码 BP（置信传播）译码器
基于 G^{-1} 冻结位约束的校验矩阵 H 上 min-sum BP，含早停
"""
import numpy as np
import math
from encoder import polar_encode, polar_generator_matrix


def _gf2_inverse(A):
    A = A.astype(np.int8) % 2
    n = A.shape[0]
    aug = np.concatenate([A.copy(), np.eye(n, dtype=np.int8)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("matrix not invertible")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _parity_from_polar(frozen_bits, N):
    Ginv = _gf2_inverse(polar_generator_matrix(N))
    frozen_idx = np.where(frozen_bits)[0]
    return Ginv[:, frozen_idx].T.astype(np.int8)


class BPDecoder:
    """min-sum BP（码字域迭代，u = x G^{-1}）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _parity_from_polar(self.frozen_bits, N)
        self.M, self.Nn = self.H.shape
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.Ginv = _gf2_inverse(polar_generator_matrix(N))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        M, N = self.M, self.Nn
        Lr = np.zeros((M, N), dtype=np.float64)
        num_iters = self.max_iter
        vn_post = llr_ch.copy()

        for it in range(1, self.max_iter + 1):
            Lq = np.zeros((M, N), dtype=np.float64)
            for m in range(M):
                for v in self.cn_edges[m]:
                    Lq[m, v] = llr_ch[v] + Lr[:, v].sum() - Lr[m, v]

            for m in range(M):
                vars_ = self.cn_edges[m]
                if len(vars_) == 0:
                    continue
                vals = Lq[m, vars_]
                signs = np.sign(vals)
                signs[signs == 0] = 1
                mags = np.abs(vals)
                for idx, v in enumerate(vars_):
                    prod_sign = np.prod(signs) * signs[idx]
                    other = np.concatenate([mags[:idx], mags[idx + 1 :]])
                    min_mag = np.min(other) if other.size else mags[idx]
                    Lr[m, v] = self.alpha * prod_sign * min_mag

            vn_post = llr_ch + Lr.sum(axis=0)
            x_hat = (vn_post < 0).astype(int)
            u_hat = (x_hat @ self.Ginv) % 2
            u_hat[self.frozen_bits] = 0

            hard = (llr_ch < 0).astype(int)
            if np.array_equal(polar_encode(u_hat), hard):
                num_iters = it
                break

        x_hat = (vn_post < 0).astype(int)
        u_hat = (x_hat @ self.Ginv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
