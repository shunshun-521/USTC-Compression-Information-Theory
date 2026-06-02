"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的因子图 min-sum BP，含早停
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode

_H_CACHE = {}
_M_CACHE = {}
_MINV_CACHE = {}


def _gf2_inverse(A):
    n = A.shape[0]
    aug = np.hstack([A.astype(np.uint8), np.eye(n, dtype=np.uint8)])
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r, col]), None)
        if piv is None:
            raise ValueError("Singular matrix over GF(2)")
        if piv != col:
            aug[[col, piv]] = aug[[piv, col]]
        for r in range(n):
            if r != col and aug[r, col]:
                aug[r] ^= aug[col]
    return aug[:, n:]


def _encoding_matrix(N):
    """由 encode(e_j) 构造线性编码矩阵 M，满足 polar_encode(u) = u @ M (mod 2)"""
    if N in _M_CACHE:
        return _M_CACHE[N]
    M = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        u = np.zeros(N, dtype=np.uint8)
        u[j] = 1
        M[j] = polar_encode(u)
    _M_CACHE[N] = M
    return M


def _inverse_encoding_matrix(N):
    if N in _MINV_CACHE:
        return _MINV_CACHE[N]
    inv = _gf2_inverse(_encoding_matrix(N))
    _MINV_CACHE[N] = inv
    return inv


def _gf2_rref(M):
    """GF(2) 行最简形"""
    M = M.copy().astype(np.uint8)
    m, n = M.shape
    pivot_row = 0
    for col in range(n):
        sel = None
        for r in range(pivot_row, m):
            if M[r, col]:
                sel = r
                break
        if sel is None:
            continue
        if sel != pivot_row:
            M[[pivot_row, sel]] = M[[sel, pivot_row]]
        for r in range(m):
            if r != pivot_row and M[r, col]:
                M[r] ^= M[pivot_row]
        pivot_row += 1
    return M, pivot_row


def _parity_check_matrix(N, info_indices):
    """由 KxN 生成矩阵 Gk 构造系统形校验矩阵 H"""
    key = (N, tuple(info_indices))
    if key in _H_CACHE:
        return _H_CACHE[key]
    Gk = _encoding_matrix(N)[info_indices, :].astype(np.uint8)
    K, Nn = Gk.shape
    G = Gk.copy()
    perm = list(range(Nn))
    row = 0
    for col in range(Nn):
        if row >= K:
            break
        pivot = next((r for r in range(row, K) if G[r, col]), None)
        if pivot is None:
            continue
        if pivot != row:
            G[[row, pivot]] = G[[pivot, row]]
        for r in range(K):
            if r != row and G[r, col]:
                G[r] ^= G[row]
        if col != row:
            G[:, [row, col]] = G[:, [col, row]]
            perm[row], perm[col] = perm[col], perm[row]
        row += 1
    P = G[:, K:]
    H_sys = np.hstack([P.T, np.eye(Nn - K, dtype=np.uint8)])
    inv_perm = np.argsort(perm)
    H = H_sys[:, inv_perm]
    _H_CACHE[key] = H
    return H


class BPDecoder:
    """基于校验矩阵 H 的 min-sum BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.H = _parity_check_matrix(N, self.info_indices)
        self.M, self.Nn = self.H.shape
        self.M_inv = _inverse_encoding_matrix(N)
        # 邻接表
        self.cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_neighbors = [np.where(self.H[:, v])[0] for v in range(self.Nn)]

    def _codeword_to_source(self, x_hat):
        u_hat = (x_hat.astype(np.uint8) @ self.M_inv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        llr_ch = np.clip(llr_ch, -30, 30)

        # 消息：L_v2c[v][m], L_c2v[m][v]
        L_v2c = {v: {m: float(llr_ch[v]) for m in self.vn_neighbors[v]} for v in range(N)}
        L_c2v = {m: {v: 0.0 for v in self.cn_neighbors[m]} for m in range(self.M)}

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 校验节点更新
            for m in range(self.M):
                nbrs = self.cn_neighbors[m]
                for v in nbrs:
                    others = [o for o in nbrs if o != v]
                    if not others:
                        L_c2v[m][v] = 0.0
                        continue
                    msgs = np.array([L_v2c[o][m] for o in others])
                    sign = np.prod(np.sign(msgs))
                    mag = np.min(np.abs(msgs))
                    L_c2v[m][v] = self.alpha * sign * mag

            # 变量节点更新
            for v in range(N):
                total = llr_ch[v] + sum(L_c2v[m][v] for m in self.vn_neighbors[v])
                for m in self.vn_neighbors[v]:
                    L_v2c[v][m] = total - L_c2v[m][v]

            post = np.zeros(N)
            for v in range(N):
                post[v] = llr_ch[v] + sum(L_c2v[m][v] for m in self.vn_neighbors[v])

            x_hat = (post < 0).astype(int)
            u_hat = self._codeword_to_source(x_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        post = np.array(
            [llr_ch[v] + sum(L_c2v[m][v] for m in self.vn_neighbors[v]) for v in range(N)]
        )
        x_hat = (post < 0).astype(int)
        u_hat = self._codeword_to_source(x_hat)
        return u_hat, num_iters
