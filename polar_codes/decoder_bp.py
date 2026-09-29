"""
极化码 BP（置信传播）译码器
在码字比特 x 上进行 LDPC 风格 min-sum BP，再经 G^{-1} 恢复 u
"""
import math

import numpy as np

from encoder import polar_encode, polar_generator_matrix


def _gf2_left_nullspace(G):
    G = np.asarray(G, dtype=np.int8) % 2
    K, N = G.shape
    A = G.copy()
    pivot_cols = []
    row = 0
    for col in range(N):
        sel = next((r for r in range(row, K) if A[r, col]), None)
        if sel is None:
            continue
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        for r in range(K):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        pivot_cols.append(col)
        row += 1
        if row == K:
            break
    free_cols = [c for c in range(N) if c not in pivot_cols]
    H = np.zeros((len(free_cols), N), dtype=np.int8)
    for idx, fc in enumerate(free_cols):
        H[idx, fc] = 1
        for i, pc in enumerate(pivot_cols[:row]):
            H[idx, pc] = A[i, fc]
    return H % 2


def _gf2_inverse(G):
    G = np.asarray(G, dtype=np.int8) % 2
    N = G.shape[0]
    A = np.concatenate([G, np.eye(N, dtype=np.int8)], axis=1)
    row = 0
    for col in range(N):
        sel = next((r for r in range(row, N) if A[r, col]), None)
        if sel is None:
            raise ValueError("matrix not invertible")
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        for r in range(N):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    return A[:, N:] % 2


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.G = polar_generator_matrix(N)
        G_info = self.G[self.info_indices]
        self.H = _gf2_left_nullspace(G_info)
        self.G_inv = _gf2_inverse(self.G)
        self._build_edges()

    def _build_edges(self):
        checks, vars_ = np.where(self.H)
        self.check_neighbors = [[] for _ in range(self.H.shape[0])]
        self.var_neighbors = [[] for _ in range(self.N)]
        for c, v in zip(checks, vars_):
            self.check_neighbors[c].append(v)
            self.var_neighbors[v].append(c)

    def _u_from_x(self, x_bits):
        u = (np.asarray(x_bits, dtype=np.int8) @ self.G_inv) % 2
        u[self.frozen_bits] = 0
        return u.astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        Lq = llr_ch.copy()
        M = len(self.check_neighbors)
        Rcv = {(c, v): 0.0 for c in range(M) for v in self.check_neighbors[c]}

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for c in range(M):
                nbrs = self.check_neighbors[c]
                msgs = [Lq[v] - Rcv[(c, v)] for v in nbrs]
                for i, v in enumerate(nbrs):
                    other = msgs[:i] + msgs[i + 1 :]
                    if not other:
                        Rcv[(c, v)] = 0.0
                        continue
                    sign = np.prod([1.0 if m >= 0 else -1.0 for m in other])
                    mag = min(abs(m) for m in other)
                    Rcv[(c, v)] = self.alpha * sign * mag

            for v in range(self.N):
                Lq[v] = llr_ch[v] + sum(Rcv[(c, v)] for c in self.var_neighbors[v])

            x_hat = (Lq < 0).astype(int)
            u_hat = self._u_from_x(x_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch) and np.array_equal(polar_encode(u_hat), x_hat):
                break

        x_hat = (Lq < 0).astype(int)
        u_hat = self._u_from_x(x_hat)
        return u_hat, num_iters
