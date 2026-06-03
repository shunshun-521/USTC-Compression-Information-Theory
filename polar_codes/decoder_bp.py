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
    """基于校验矩阵 H 的 min-sum BP 译码器（向量化边消息）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.H = _parity_check_matrix(N, self.info_indices)
        self.M, self.Nn = self.H.shape
        self.M_inv = _inverse_encoding_matrix(N)
        self._build_edge_tables()

    def _build_edge_tables(self):
        cn_nbrs = [np.where(self.H[m])[0] for m in range(self.M)]
        vn_nbrs = [np.where(self.H[:, v])[0] for v in range(self.Nn)]
        edge_vn, edge_cn, eid = [], [], 0
        vn_edge_lists = [[] for _ in range(self.Nn)]
        cn_edge_lists = [[] for _ in range(self.M)]
        for m in range(self.M):
            for v in cn_nbrs[m]:
                edge_vn.append(v)
                edge_cn.append(m)
                vn_edge_lists[v].append(eid)
                cn_edge_lists[m].append(eid)
                eid += 1
        self.num_edges = eid
        self.edge_vn = np.array(edge_vn, dtype=np.int32)
        self.edge_cn = np.array(edge_cn, dtype=np.int32)
        self.vn_edge_lists = [np.array(x, dtype=np.int32) for x in vn_edge_lists]
        self.cn_edge_lists = [np.array(x, dtype=np.int32) for x in cn_edge_lists]

    def _codeword_to_source(self, x_hat):
        u_hat = (x_hat.astype(np.uint8) @ self.M_inv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -30, 30)
        N = self.N
        v2c = np.zeros(self.num_edges, dtype=np.float64)
        c2v = np.zeros(self.num_edges, dtype=np.float64)
        for v in range(N):
            v2c[self.vn_edge_lists[v]] = llr_ch[v]

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for m in range(self.M):
                edges = self.cn_edge_lists[m]
                msgs = v2c[edges]
                d = len(msgs)
                if d == 1:
                    c2v[edges[0]] = 0.0
                    continue
                signs = np.sign(msgs)
                mags = np.abs(msgs)
                total_sign = np.prod(signs)
                idx_min = int(np.argmin(mags))
                min1 = mags[idx_min]
                min2 = np.min(np.delete(mags, idx_min)) if d > 1 else 0.0
                for i, e in enumerate(edges):
                    mag = min2 if i == idx_min else min1
                    c2v[e] = self.alpha * total_sign * signs[i] * mag

            for v in range(N):
                edges = self.vn_edge_lists[v]
                total = llr_ch[v] + np.sum(c2v[edges])
                v2c[edges] = total - c2v[edges]

            post = np.zeros(N, dtype=np.float64)
            for v in range(N):
                post[v] = llr_ch[v] + np.sum(c2v[self.vn_edge_lists[v]])

            x_hat = (post < 0).astype(int)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break
        else:
            num_iters = self.max_iter

        x_hat = (post < 0).astype(int)
        return self._codeword_to_source(x_hat), num_iters
