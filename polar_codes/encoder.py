"""
极化码编码器
编码：x = u * G_N，G_N = F^（n 次 Kronecker）（行向量乘积，GF(2)）
"""
import numpy as np

_F_CACHE = {}


def _build_F(n):
    if n not in _F_CACHE:
        G = np.eye(1, dtype=np.int8)
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        for _ in range(n):
            G = np.kron(G, F)
        _F_CACHE[n] = G.astype(np.int8)
    return _F_CACHE[n]


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    perm = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        perm[i] = r
    return perm


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，x = (u @ F^n) mod 2
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    G = _build_F(n)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode N=4:", polar_encode(u))
