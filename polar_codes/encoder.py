"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
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
    极化码编码：蝶形 XOR（与 F^{⊗n} 左乘一致），输出含比特倒序置换 B_N。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be power of 2")
    n = int(np.log2(N))
    block = N
    while block >= 2:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                u[start + k] ^= u[start + k + half]
        block = half
    br = bit_reversal_permutation(N)
    return u[br].astype(int)


def polar_encode_matrix(u):
    """GF(2) 矩阵乘法验证用"""
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return (u @ G) % 2
