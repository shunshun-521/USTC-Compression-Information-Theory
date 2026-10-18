"""
极化码编码器
编码：x = u * F^{\\otimes n}（大块优先蝶形 XOR，O(N log N)）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.empty(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        indices[i] = r
    return indices


def bit_reversed(x, n):
    """单整数比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（大块优先、原地 XOR，与 SC 因子图一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                idx = p + k
                u[idx] ^= u[idx + n_split]
        n = n_split
    return u.astype(np.int8)


def polar_encode_with_br(u):
    """x = u @ (B_N F^{\\otimes n})，用于与生成矩阵定义对照。"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    for _ in range(n):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    GN = (np.eye(N, dtype=np.int8)[br] @ G) % 2
    return (u @ GN) % 2
