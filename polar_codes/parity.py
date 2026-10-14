"""极化码校验矩阵构造与基于校验矩阵的 BP 译码。"""
import numpy as np
from functools import lru_cache
from encoder import build_generator_matrix


def gf2_nullspace_left(G_eff):
    """求 H 行向量，满足 H @ G_eff^T = 0 (mod 2)，G_eff 形状 (K, N)。"""
    A = (G_eff.T % 2).astype(np.uint8)
    m, n = A.shape
    aug = np.hstack([A, np.eye(m, dtype=np.uint8)])
    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if aug[r, col]:
                piv = r
                break
        if piv is None:
            continue
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        for r in range(m):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    if row == m:
        return np.zeros((0, m), dtype=np.uint8)
    return np.array([aug[r, n:] for r in range(row, m)], dtype=np.uint8)


@lru_cache(maxsize=16)
def get_parity_matrix(N, info_tuple):
    """由信息位索引构造 (N-K) x N 校验矩阵。"""
    info = np.array(info_tuple, dtype=int)
    G = build_generator_matrix(N)
    G_eff = G[info, :]
    return gf2_nullspace_left(G_eff)


def minsum_bp_decode(H, llr_ch, max_iter=60):
    """
    标准 min-sum BP（校验矩阵 H），输入信道 LLR，输出码字硬判决 x_hat。
    """
    H = np.asarray(H, dtype=np.uint8)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    m, n = H.shape
    if m == 0:
        return (llr_ch < 0).astype(int)

    cn_neighbors = [np.where(H[i])[0] for i in range(m)]
    vn_neighbors = [np.where(H[:, j])[0] for j in range(n)]

    Lq = np.zeros((m, n), dtype=np.float64)
    Rcv = np.zeros((m, n), dtype=np.float64)

    for it in range(max_iter):
        for i in range(m):
            nbr = cn_neighbors[i]
            for j in nbr:
                sgn = 1.0
                minm = 1e30
                for k in nbr:
                    if k == j:
                        continue
                    v = Lq[i, k] - Rcv[i, k]
                    sgn *= np.sign(v) if v != 0 else 1.0
                    minm = min(minm, abs(v))
                Rcv[i, j] = 0.75 * sgn * minm

        Lpost = llr_ch.copy()
        for j in range(n):
            Lpost[j] += Rcv[vn_neighbors[j], j].sum()

        x_hat = (Lpost < 0).astype(int)
        if np.all((H @ x_hat) % 2 == 0):
            return x_hat

        for i in range(m):
            for j in cn_neighbors[i]:
                Lq[i, j] = Lpost[j] - Rcv[i, j]

    Lpost = llr_ch + Rcv.sum(axis=0)
    return (Lpost < 0).astype(int)
