"""
极化码 BP（置信传播）译码器
在极化码校验矩阵上执行 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, polar_generator_matrix


def _gf2_inverse(G):
    G = np.asarray(G, dtype=np.uint8).copy() % 2
    N = G.shape[0]
    aug = np.hstack([G, np.eye(N, dtype=np.uint8)])
    row = 0
    for col in range(N):
        sel = None
        for r in range(row, N):
            if aug[r, col]:
                sel = r
                break
        if sel is None:
            raise ValueError("Matrix is singular in GF(2)")
        if sel != row:
            aug[[row, sel]] = aug[[sel, row]]
        for r in range(N):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, N:]


def _gf2_nullspace_rows(G_info):
    """
    给定 G_info (K x N)，返回 H ((N-K) x N) 使 G_info @ h = 0 对所有列 h 成立。
    """
    G = np.asarray(G_info, dtype=np.uint8).copy() % 2
    K, N = G.shape
    pivots = []
    row = 0
    for col in range(N):
        sel = None
        for r in range(row, K):
            if G[r, col]:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            G[[row, sel]] = G[[sel, row]]
        pivots.append(col)
        for r in range(K):
            if r != row and G[r, col]:
                G[r] ^= G[row]
        row += 1

    free_cols = [c for c in range(N) if c not in pivots]
    basis = []
    for fc in free_cols:
        v = np.zeros(N, dtype=np.uint8)
        v[fc] = 1
        for i, pc in enumerate(pivots):
            if G[i, fc]:
                v[pc] = 1
        basis.append(v)
    if not basis:
        return np.zeros((0, N), dtype=np.uint8)
    return np.array(basis, dtype=np.uint8)


class BPDecoder:
    """BP 译码器（校验矩阵 min-sum BP）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        info_idx = np.where(~self.frozen_bits)[0]
        G = polar_generator_matrix(N)
        G_info = G[info_idx, :]
        self.H = _gf2_nullspace_rows(G_info)
        self.G_inv = _gf2_inverse(G)
        self.M, self.N_code = self.H.shape
        self._cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self._vn_edges = [np.where(self.H[:, v])[0] for v in range(self.N_code)]

    def _f_ms(self, msgs):
        msgs = [self.alpha * m for m in msgs if abs(m) < self.LARGE * 0.5]
        if not msgs:
            return 0.0
        sign = np.prod(np.sign(msgs))
        mag = min(abs(m) for m in msgs)
        return sign * mag

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        ch_llr = llr_raw.copy()
        N = self.N
        M = self.M

        Lq = ch_llr.copy()
        Lr = np.zeros((M, N), dtype=np.float64)

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for m in range(M):
                idx = self._cn_edges[m]
                for v in idx:
                    others = [Lq[j] + Lr[m, j] for j in idx if j != v]
                    Lr[m, v] = self._f_ms(others)

            for v in range(N):
                Lq[v] = ch_llr[v] + np.sum(Lr[self._vn_edges[v], v])

            x_hat = (Lq < 0).astype(int)
            u_hat = (x_hat @ self.G_inv) % 2
            u_hat[self.frozen_bits] = 0

            x_reenc = polar_encode(u_hat)
            hard = (llr_raw < 0).astype(int)
            if np.array_equal(x_reenc, hard):
                return u_hat, num_iters

        x_hat = (Lq < 0).astype(int)
        u_hat = (x_hat @ self.G_inv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
