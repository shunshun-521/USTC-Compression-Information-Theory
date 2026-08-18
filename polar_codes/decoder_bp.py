"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停机制
"""
import numpy as np
from numba import njit

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


@njit(cache=True)
def _f_mm(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


@njit(cache=True)
def _check_node_minsum(msgs, alpha):
    d = len(msgs)
    if d == 0:
        return msgs
    if d == 1:
        out = np.zeros(1)
        return out
    F = np.empty(d)
    B = np.empty(d)
    F[0] = msgs[0]
    for i in range(1, d):
        F[i] = _f_mm(F[i - 1], msgs[i], alpha)
    B[d - 1] = msgs[d - 1]
    for i in range(d - 2, -1, -1):
        B[i] = _f_mm(B[i + 1], msgs[i + 1], alpha)
    out = np.empty(d)
    out[0] = B[1]
    out[d - 1] = F[d - 2]
    for i in range(1, d - 1):
        out[i] = _f_mm(F[i - 1], B[i + 1], alpha)
    return out


@njit(cache=True)
def _bp_decode_numba(llr_ch, check_vars, check_deg, max_iter, alpha):
    N = len(llr_ch)
    M = check_vars.shape[0]
    max_d = check_vars.shape[1]
    Lv = llr_ch.copy()
    Rcv = np.zeros((M, N))
    hard = (llr_ch < 0).astype(np.int8)
    num_iters = 0

    for it in range(max_iter):
        num_iters = it + 1
        for c in range(M):
            d = check_deg[c]
            msgs = np.empty(d)
            for j in range(d):
                v = check_vars[c, j]
                msgs[j] = Lv[v] - Rcv[c, v]
            outs = _check_node_minsum(msgs, alpha)
            for j in range(d):
                v = check_vars[c, j]
                Rcv[c, v] = outs[j]

        for v in range(N):
            s = llr_ch[v]
            for c in range(M):
                s += Rcv[c, v]
            Lv[v] = s

        mismatch = False
        for v in range(N):
            x = 1 if Lv[v] < 0 else 0
            if x != hard[v]:
                mismatch = True
                break
        if not mismatch:
            break

    return Rcv, num_iters


class BPDecoder:
    """基于校验矩阵的 BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

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
        self.M = H.shape[0]
        check_to_vars = [np.where(H[c])[0] for c in range(self.M)]
        max_d = max(len(v) for v in check_to_vars) if self.M > 0 else 0
        check_vars = np.full((self.M, max_d), -1, dtype=np.int32)
        check_deg = np.zeros(self.M, dtype=np.int32)
        for c, vars_idx in enumerate(check_to_vars):
            check_deg[c] = len(vars_idx)
            check_vars[c, :len(vars_idx)] = vars_idx

        self._check_vars = check_vars
        self._check_deg = check_deg

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        Rcv, num_iters = _bp_decode_numba(
            llr_ch, self._check_vars, self._check_deg, self.max_iter, self.alpha
        )
        total = llr_ch + np.sum(Rcv, axis=0)
        x_hat = (total < 0).astype(int)
        u_hat = polar_encode(x_hat.copy())
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
