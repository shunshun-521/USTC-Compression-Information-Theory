"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停
"""
import math
import numpy as np

from encoder import polar_encode


def _build_polar_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    G = F.copy()
    n = int(math.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G.astype(np.uint8)


def _null_space_gf2(A):
    """A: m x n，返回 (basis, dim) 其中 basis shape (dim, n)。"""
    A = A.copy().astype(np.uint8)
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        sel = None
        for r in range(row, m):
            if A[r, col]:
                sel = r
                break
        if sel is None:
            continue
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        pivots.append(col)
        row += 1
    free_cols = [c for c in range(n) if c not in pivots]
    basis = []
    for fc in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[fc] = 1
        for j, pc in enumerate(pivots):
            if A[j, fc]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8), len(free_cols)


def _parity_check_matrix(N, frozen_bits):
    """构造 (N-K) x N 校验矩阵 H（GF2），满足 H @ x^T = 0 对所有合法码字 x。"""
    G = _build_polar_G(N)
    info_idx = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
    K = len(info_idx)
    G_rows = G[info_idx, :]  # K x N，x = u_info @ G_rows
    null_basis, _ = _null_space_gf2(G_rows)
    return null_basis.astype(np.uint8), info_idx


class BPDecoder:
    """Min-sum BP 译码器（基于 H 矩阵）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H, self.info_idx = _parity_check_matrix(N, frozen_bits)
        self.M, self.N = self.H.shape
        self._cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self._vn_neighbors = [np.where(self.H[:, v])[0] for v in range(self.N)]

    def _f_ms(self, a, b):
        sa = np.sign(a)
        sb = np.sign(b)
        mag = np.minimum(np.abs(a), np.abs(b))
        return self.alpha * sa * sb * mag

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.M

        Lq = llr_ch.copy()
        Lr = np.zeros((M, N), dtype=np.float64)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 校验节点更新
            for m in range(M):
                nbrs = self._cn_neighbors[m]
                for v in nbrs:
                    prod_sign = 1.0
                    min_mag = np.inf
                    for v2 in nbrs:
                        if v2 == v:
                            continue
                        q = Lq[v2] - Lr[m, v2]
                        prod_sign *= np.sign(q) if q != 0 else 1.0
                        min_mag = min(min_mag, np.abs(q))
                    Lr[m, v] = self.alpha * prod_sign * min_mag

            # 变量节点更新
            for v in range(N):
                Lq[v] = llr_ch[v] + np.sum(Lr[:, v])

            x_hat = (Lq < 0).astype(int)
            u_hat = polar_encode(x_hat)
            u_hat[self.frozen_bits] = 0

            syndrome = (self.H @ x_hat) % 2
            num_iters = it
            if not np.any(syndrome):
                break

        return u_hat, num_iters
