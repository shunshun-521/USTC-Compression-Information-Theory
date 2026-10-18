"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（与 Arikan 核 F=[[1,1],[0,1]] 的 Kronecker 积一致）
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def arikan_generator_matrix(N):
    """G = F^{\\otimes n}, F=[[1,1],[0,1]]"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    n = int(np.log2(N))
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    非递归极化编码（与 arikan_generator_matrix @ u 等价）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    n_blk = N
    while n_blk > 1:
        n_split = n_blk // 2
        for p in range(0, N, n_blk):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_blk = n_split
    return (u % 2).astype(int)


def polar_encode_matrix(u):
    """矩阵形式编码（校验用）"""
    u = np.asarray(u, dtype=np.int8)
    G = arikan_generator_matrix(len(u))
    return (G @ u) % 2
