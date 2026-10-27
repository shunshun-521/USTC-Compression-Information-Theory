"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx & 1) << (n - 1))
    for bit in range(1, n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    蝶形：相邻对 (u[i], u[i+step]) -> (u[i] XOR u[i+step], u[i+step])
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            u[i : i + step] ^= u[i + step : i + 2 * step]
    rev = bit_reversal_permutation(N)
    return u[rev].astype(np.int8)


def polar_encode_matrix(N):
    """生成 G_N = B_N F^{⊗n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Fn = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        Fn = np.kron(Fn, F)
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    B[rev, np.arange(N)] = 1
    return (B @ Fn) % 2
