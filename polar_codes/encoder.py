"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64
    )


def bit_reversed_index(i, n):
    """单索引比特倒序（与 mcba1n polar-codes 一致）"""
    result = 0
    for b in range(n):
        if i & (1 << b):
            result |= 1 << (n - 1 - b)
    return result


def polar_encode(u):
    """
    极化码非系统编码（大区块优先蝶形 XOR，无输出比特倒序）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    while n_block > 1:
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_block = n_split
    return u.astype(np.int8)


def polar_encode_matrix(N):
    """生成 G_N 用于校验（GF(2)）"""
    G = np.eye(N, dtype=np.int8)
    for i in range(N):
        G[i] = polar_encode(G[i])
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode", u, "->", x)
