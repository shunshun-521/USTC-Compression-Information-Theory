"""GF(2) 线性代数工具（用于与生成矩阵一致的硬判决译码）。"""
import numpy as np

_GINV_CACHE = {}


def gf2_mat_inv(G):
    N = G.shape[0]
    A = np.hstack([G.copy(), np.eye(N, dtype=int)]) % 2
    for col in range(N):
        pivot = None
        for row in range(col, N):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("GF(2) matrix is singular")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
        for row in range(N):
            if row != col and A[row, col] == 1:
                A[row] = (A[row] + A[col]) % 2
    return A[:, N:]


def gf2_decode_from_llr(llr, G_inv):
    """从信道 LLR 硬切片后解 u = x G^{-1} (mod 2)。"""
    x_hat = (np.asarray(llr, dtype=np.float64) < 0).astype(int)
    return (x_hat @ G_inv) % 2


def get_ginv(N, G):
    if N not in _GINV_CACHE:
        _GINV_CACHE[N] = gf2_mat_inv(G)
    return _GINV_CACHE[N]
