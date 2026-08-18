"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _gf2_nullspace(G):
    """求 GF(2) 上 G 的零空间基（行向量 x 满足 x @ G == 0）"""
    G = np.array(G, dtype=np.uint8)
    m, n = G.shape
    A = G.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        pivot_cols.append(col)
        for r in range(m):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[fc] = 1
        for i, pc in enumerate(pivot_cols):
            if A[i, fc]:
                vec[pc] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8)


class BPDecoder:
    """基于校验矩阵的 BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

        info_idx = np.where(~self.frozen_bits)[0]
        K = len(info_idx)
        G_info = np.zeros((N, K), dtype=np.uint8)
        for k, idx in enumerate(info_idx):
            u = np.zeros(N, dtype=int)
            u[idx] = 1
            G_info[:, k] = polar_encode(u)

        H = _gf2_nullspace(G_info.T)
        if H.ndim == 1:
            H = H.reshape(0, N)
        self.H = H
        self.M = H.shape[0]
        self.var_to_checks = [np.where(H[:, v])[0] for v in range(N)]
        self.check_to_vars = [np.where(H[c])[0] for c in range(self.M)]

    def _f_minsum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.M

        Lv = llr_ch.copy()
        Rcv = np.zeros((M, N), dtype=np.float64)

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for c in range(M):
                vars_idx = self.check_to_vars[c]
                msgs = [Lv[v] - Rcv[c, v] for v in vars_idx]
                for i, v in enumerate(vars_idx):
                    others = [msgs[j] for j in range(len(vars_idx)) if j != i]
                    if not others:
                        prod = 0.0
                    else:
                        prod = others[0]
                        for o in others[1:]:
                            prod = self._f_minsum(prod, o)
                    Rcv[c, v] = prod

            total = llr_ch + np.sum(Rcv, axis=0)
            Lv = total.copy()
            x_hat = (total < 0).astype(int)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = llr_ch + np.sum(Rcv, axis=0)
        x_hat = (total < 0).astype(int)
        u_hat = polar_encode(x_hat.copy())
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
