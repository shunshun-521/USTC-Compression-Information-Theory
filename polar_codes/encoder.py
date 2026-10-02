"""
极化码编码器
编码：x = G @ u（蝶形结构 O(N log N)）
"""
import numpy as np

_G_CACHE = {}
_GINV_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形结构，与因子图一致的码字顺序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i] ^= u[i + step]
        step <<= 1
    return u


def _gf2_inverse(G):
    N = G.shape[0]
    A = np.concatenate([G.copy(), np.eye(N, dtype=int)], axis=1)
    row = 0
    for col in range(N):
        pivot = None
        for r in range(row, N):
            if A[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(N):
            if r != row and A[r, col]:
                A[r] ^= A[row]
        row += 1
    return A[:, N:]


def get_generator_matrix(N):
    """返回 N×N 生成矩阵 G（列向量为单位源向量的编码结果）"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[:, i] = polar_encode(e)
    _G_CACHE[N] = G
    return G


def get_inverse_generator_matrix(N):
    """返回 GF(2) 上的 G^{-1}，满足 G @ G^{-1} = I"""
    if N in _GINV_CACHE:
        return _GINV_CACHE[N]
    G = get_generator_matrix(N)
    Ginv = _gf2_inverse(G).astype(np.int8)
    _GINV_CACHE[N] = Ginv
    return Ginv


def polar_generator_matrix(N):
    """G_N = B_N F^{\\otimes n}（行置换形式，供对照）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return G[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    G = get_generator_matrix(4)
    print("G@u mod2:", np.mod(G @ u, 2))
